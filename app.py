import os
import sys
import json
import logging
import torch
import random
import numpy as np
import requests
from dataclasses import dataclass, field
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import Dataset


@dataclass
class Config:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    teacher_api_url: str = "http://localhost:1234/v1/chat/completions"
    teacher_model_name: str = "qwen/qwen3-4b-2507"
    save_dir: str = "./student_model"
    chat_history_file: str = field(init=False)
    batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 5e-5
    train_buffer_size: int = 4

    def __post_init__(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.chat_history_file = os.path.join(self.save_dir, "chat_history.json")


class DialogueDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128):  # Уменьшено до 128 для экономии памяти
        self.examples = []
        for text in texts:
            enc = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=block_size,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"][0]
            attn_mask = enc["attention_mask"][0]
            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "labels": input_ids.clone()
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class SimpleChatAGI:
    def __init__(self):
        self.cfg = Config()
        self.logger = self.setup_logging()

        # Инициализация токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Явное указание устройства (используем GPU 0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Используемое устройство: {self.device}")

        # Загрузка модели на конкретное устройство
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()  # Начинаем в режиме инференса

        self.buffer = []
        self.chat_history = self.load_chat_history()
        self.training_step = 0

    def setup_logging(self):
        logger = logging.getLogger("chat_agi")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        fh = logging.FileHandler(os.path.join(self.cfg.save_dir, "chat_agi.log"), encoding="utf-8")
        fh.setFormatter(formatter)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    def load_chat_history(self):
        if os.path.exists(self.cfg.chat_history_file):
            try:
                with open(self.cfg.chat_history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Ошибка загрузки истории: {e}")
        return []

    def save_chat_history(self):
        try:
            with open(self.cfg.chat_history_file, "w", encoding="utf-8") as f:
                json.dump(self.chat_history[-200:], f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения истории: {e}")

    def add_to_history(self, role, content):
        self.chat_history.append({
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content
        })
        self.save_chat_history()

    def query_teacher_model(self, user_input):
        try:
            payload = {
                "model": self.cfg.teacher_model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that provides high-quality responses for training other models."
                    },
                    {"role": "user", "content": user_input}
                ],
                "temperature": 0.7,
                "max_tokens": 512,
                "stream": False
            }

            response = requests.post(self.cfg.teacher_api_url, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                self.logger.error(f"Ошибка API учителя: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            self.logger.error(f"Ошибка запроса к учителю: {e}")
            return None

    def generate(self, prompt, max_new_tokens=150):
        self.model.eval()  # Убедимся, что в режиме инференса
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
                add_special_tokens=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_k=40,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated = outputs[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            return text.strip()
        except Exception as e:
            self.logger.error(f"Ошибка генерации: {e}")
            return "Извините, произошла ошибка при генерации ответа."

    def train_on_buffer(self):
        if len(self.buffer) < self.cfg.train_buffer_size:
            self.logger.info(f"Буфер обучения: {len(self.buffer)}/{self.cfg.train_buffer_size}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False

        self.training_step += 1
        self.logger.info(f"🚀 Начало обучения шаг {self.training_step} на {len(self.buffer)} примерах...")

        try:
            self.model.train()

            dataset = DialogueDataset(self.buffer, self.tokenizer)
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            training_args = TrainingArguments(
                output_dir=os.path.join(self.cfg.save_dir, f"checkpoint_{self.training_step}"),
                overwrite_output_dir=True,
                num_train_epochs=self.cfg.epochs,
                per_device_train_batch_size=self.cfg.batch_size,
                learning_rate=self.cfg.learning_rate,
                logging_steps=1,
                save_strategy="no",
                fp16=False,
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                report_to=[],
                logging_dir="./logs",
                no_cuda=False,
                local_rank=-1,
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )

            train_result = trainer.train()

            train_loss = train_result.metrics.get('train_loss', 'N/A')
            self.logger.info(f"✅ Обучение завершено! Loss: {train_loss}")

            # Сохраняем модель
            self.model.save_pretrained(self.cfg.save_dir)
            self.tokenizer.save_pretrained(self.cfg.save_dir)
            self.logger.info(f"💾 Модель сохранена в {self.cfg.save_dir}")

            # 🔥 КРИТИЧЕСКИ ВАЖНО: освобождение памяти
            del trainer
            del dataset
            del data_collator

            self.model.eval()
            self.buffer.clear()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True

        except Exception as e:
            self.logger.error(f"❌ Ошибка обучения: {e}")
            try:
                self.model.save_pretrained(self.cfg.save_dir + "_backup")
                self.logger.info("💾 Создана резервная копия модели")
            except Exception as save_err:
                self.logger.error(f"Не удалось создать резервную копию: {save_err}")

            self.model.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False

    def chat_loop(self):
        print("=" * 60)
        print("🤖 Chat-AGI — обучение с помощью Qwen3-4B-2507!")
        print(f"💻 Модель: {self.cfg.model_name}")
        print(f"💾 Сохранение в: {self.cfg.save_dir}")
        print(f"📊 Размер буфера: {self.cfg.train_buffer_size}")
        if torch.cuda.is_available():
            print(f"🔢 Доступно GPU: {torch.cuda.device_count()}")
            print(f"🎯 Используется: {self.device}")
        print("=" * 60)
        print("Команды: exit/quit/выход, save, clear, status")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n👤 Пользователь: ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", "выход"):
                    self.save_chat_history()
                    self.model.save_pretrained(self.cfg.save_dir)
                    print("💾 Сохранено. До свидания!")
                    break
                elif user_input.lower() == "save":
                    self.model.save_pretrained(self.cfg.save_dir)
                    print("✅ Модель сохранена")
                    continue
                elif user_input.lower() == "clear":
                    self.buffer.clear()
                    print("✅ Буфер очищен")
                    continue
                elif user_input.lower() == "status":
                    print(f"📊 Статус: буфер {len(self.buffer)}/{self.cfg.train_buffer_size}")
                    if torch.cuda.is_available():
                        mem = torch.cuda.memory_allocated(self.device) / 1e9
                        print(f"GMEM: {mem:.2f} GB")
                    continue

                self.add_to_history("user", user_input)

                student_prompt = f"### Пользователь: {user_input}\n### Ассистент:"
                student_response = self.generate(student_prompt)
                print(f"🤖 СТУДЕНТ: {student_response}")

                print("🔄 Запрос к учителю...")
                teacher_response = self.query_teacher_model(user_input)

                if teacher_response:
                    print(f"👨🏫 УЧИТЕЛЬ: {teacher_response}")
                    training_response = teacher_response
                else:
                    print("⚠️ Учитель недоступен, используем ответ студента")
                    training_response = student_response

                self.add_to_history("assistant", training_response)
                formatted_dialogue = f"### Пользователь: {user_input}\n### Ассистент: {training_response}</s>"
                self.buffer.append(formatted_dialogue)

                print(f"📥 Добавлено в буфер: {len(self.buffer)}/{self.cfg.train_buffer_size}")

                if self.train_on_buffer():
                    print("🎉 Обучение завершено успешно!")
                    print("🔄 Тестирование после обучения...")
                    test_response = self.generate(student_prompt)
                    print(f"🤖 СТУДЕНТ (после обучения): {test_response}")

            except KeyboardInterrupt:
                print("\n\n⚠️ Прервано пользователем")
                self.save_chat_history()
                break
            except Exception as e:
                self.logger.error(f"Ошибка в цикле чата: {e}")
                print(f"❌ Произошла ошибка: {e}")
                continue


if __name__ == "__main__":
    # Указываем, что используем только GPU 0 (даже если их два)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    chat_agi = SimpleChatAGI()
    chat_agi.chat_loop()