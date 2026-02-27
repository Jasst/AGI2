#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 САМООБУЧАЮЩИЙСЯ МИНИ-МОЗГ БОТ v10.0 - УЛУЧШЕННАЯ ВЕРСИЯ
✅ ИСПРАВЛЕНА логика веб-поиска с интеллектуальным анализом
✅ ДОБАВЛЕНО расширенное воображение и синтез информации
✅ УЛУЧШЕНА обработка результатов поиска через LLM
✅ ДОБАВЛЕНА система creative synthesis для генерации новых идей
✅ УЛУЧШЕНА интеграция между поиском и памятью
"""

import os
import json
import re
import asyncio
import aiohttp
import traceback
import hashlib
import math
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# Импорт для парсинга веб-страниц
try:
    from bs4 import BeautifulSoup

    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("⚠️ BeautifulSoup не установлен. Парсинг веб-страниц будет ограничен.")

# DuckDuckGo Search (поддержка Python 3.13)
DDGS_AVAILABLE = False
DDGS_ASYNC = False
try:
    from duckduckgo_search import AsyncDDGS

    DDGS_AVAILABLE = True
    DDGS_ASYNC = True
    print("✅ Async DuckDuckGo Search доступен")
except ImportError:
    try:
        from duckduckgo_search import DDGS

        DDGS_AVAILABLE = True
        DDGS_ASYNC = False
        print("✅ DuckDuckGo Search доступен (синхронный режим)")
    except ImportError:
        print("⚠️ DuckDuckGo Search не установлен")

# ==================== КОНФИГУРАЦИЯ ====================
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
if not TELEGRAM_TOKEN:
    raise ValueError("❌ ОШИБКА: Не найден TELEGRAM_TOKEN в .env!")

LM_STUDIO_API_URL = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
LM_STUDIO_API_KEY = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')

# Директории
CORES_DIR = "dynamic_cores"
MEMORY_DIR = "brain_memory"
USER_FILES_DIR = "user_files"
CACHE_DIR = "cache"

for directory in [CORES_DIR, MEMORY_DIR, USER_FILES_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Файлы
LEARNING_LOG = os.path.join(MEMORY_DIR, "learning_log.json")
CORE_PERFORMANCE_LOG = os.path.join(MEMORY_DIR, "core_performance.json")
WEB_CACHE_FILE = os.path.join(CACHE_DIR, "web_search_cache.json")


# ==================== БАЗОВЫЕ УТИЛИТЫ ====================
class FileManager:
    """Надёжное управление файлами с атомарной записью"""

    @staticmethod
    def safe_save_json(filepath: str, data: Any) -> bool:
        """Безопасное сохранение JSON с атомарной записью"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            temp_file = f"{filepath}.tmp"

            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            if os.path.exists(filepath):
                backup_file = f"{filepath}.backup"
                os.replace(filepath, backup_file)

            os.replace(temp_file, filepath)
            return True
        except Exception as e:
            print(f"⚠️ Ошибка сохранения {filepath}: {e}")
            backup_file = f"{filepath}.backup"
            if os.path.exists(backup_file) and not os.path.exists(filepath):
                try:
                    os.replace(backup_file, filepath)
                except:
                    pass
            return False

    @staticmethod
    def safe_load_json(filepath: str, default: Any = None) -> Any:
        """Безопасная загрузка JSON с восстановлением из backup"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️ Повреждённый JSON {filepath}, пытаюсь восстановить из backup: {e}")
            backup_file = f"{filepath}.backup"
            if os.path.exists(backup_file):
                try:
                    with open(backup_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        print(f"✅ Восстановлено из backup")
                        return data
                except:
                    pass
        except Exception as e:
            print(f"⚠️ Ошибка загрузки {filepath}: {e}")

        return default


class TimeUtils:
    """Улучшенная работа со временем"""

    MONTHS_RU = {
        'январ': 1, 'феврал': 2, 'март': 3, 'апрел': 4,
        'ма': 5, 'июн': 6, 'июл': 7, 'август': 8,
        'сентябр': 9, 'октябр': 10, 'ноябр': 11, 'декабр': 12
    }

    DAYS_RU = {
        'понедельник': 0, 'вторник': 1, 'среда': 2, 'четверг': 3,
        'пятница': 4, 'суббота': 5, 'воскресенье': 6
    }

    @staticmethod
    def get_current_time_info() -> Dict[str, Any]:
        """Получение полной информации о текущем времени"""
        now = datetime.now()
        return {
            'datetime': now,
            'date': now.date(),
            'time': now.time(),
            'timestamp': now.timestamp(),
            'iso': now.isoformat(),
            'formatted': now.strftime('%d.%m.%Y %H:%M:%S'),
            'year': now.year,
            'month': now.month,
            'day': now.day,
            'hour': now.hour,
            'minute': now.minute,
            'second': now.second,
            'weekday': now.weekday(),
            'weekday_name': now.strftime('%A'),
            'month_name': now.strftime('%B'),
            'day_of_year': now.timetuple().tm_yday
        }

    @staticmethod
    def parse_relative_time(text: str) -> Optional[datetime]:
        """Парсинг относительного времени"""
        text_lower = text.lower()
        now = datetime.now()

        if 'сегодня' in text_lower:
            return now
        if 'вчера' in text_lower:
            return now - timedelta(days=1)
        if 'завтра' in text_lower:
            return now + timedelta(days=1)
        if 'послезавтра' in text_lower:
            return now + timedelta(days=2)

        through_match = re.search(r'через\s+(\d+)\s+(день|дня|дней|час|часа|часов|минут)', text_lower)
        if through_match:
            amount = int(through_match.group(1))
            unit = through_match.group(2)

            if 'день' in unit or 'дня' in unit or 'дней' in unit:
                return now + timedelta(days=amount)
            elif 'час' in unit:
                return now + timedelta(hours=amount)
            elif 'минут' in unit:
                return now + timedelta(minutes=amount)

        ago_match = re.search(r'(\d+)\s+(день|дня|дней|час|часа|часов|минут)\s+назад', text_lower)
        if ago_match:
            amount = int(ago_match.group(1))
            unit = ago_match.group(2)

            if 'день' in unit or 'дня' in unit or 'дней' in unit:
                return now - timedelta(days=amount)
            elif 'час' in unit:
                return now - timedelta(hours=amount)
            elif 'минут' in unit:
                return now - timedelta(minutes=amount)

        return None

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Форматирование длительности"""
        if seconds < 60:
            return f"{seconds:.1f} сек"
        elif seconds < 3600:
            return f"{seconds / 60:.1f} мин"
        elif seconds < 86400:
            return f"{seconds / 3600:.1f} ч"
        else:
            return f"{seconds / 86400:.1f} дн"

    @staticmethod
    def get_time_of_day() -> str:
        """Определение времени суток"""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "утро"
        elif 12 <= hour < 17:
            return "день"
        elif 17 <= hour < 22:
            return "вечер"
        else:
            return "ночь"


class MemoryUtils:
    """Утилиты для работы с памятью"""

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Вычисление схожести текстов"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def extract_keywords(text: str, top_n: int = 5) -> List[str]:
        """Извлечение ключевых слов"""
        stop_words = {'в', 'и', 'на', 'с', 'по', 'для', 'от', 'к', 'о', 'у', 'из', 'за', 'что', 'это', 'как', 'то', 'а',
                      'но', 'или', 'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were', 'been',
                      'be'}
        words = [w.lower() for w in re.findall(r'\b\w{3,}\b', text)]
        filtered = [w for w in words if w not in stop_words]

        counter = Counter(filtered)
        return [word for word, _ in counter.most_common(top_n)]

    @staticmethod
    def calculate_importance(entry: Dict[str, Any]) -> float:
        """Вычисление важности записи памяти"""
        importance = 0.5

        if entry.get('has_question'):
            importance += 0.1
        if entry.get('has_numbers'):
            importance += 0.1
        if len(entry.get('extracted_facts', [])) > 0:
            importance += 0.2
        if len(entry.get('detected_emotions', [])) > 0:
            importance += 0.1
        if entry.get('word_count', 0) > 50:
            importance += 0.1

        return min(1.0, importance)


# ==================== УЛУЧШЕННЫЙ ВЕБ-ПОИСК С ИНТЕЛЛЕКТУАЛЬНЫМ АНАЛИЗОМ ====================
class EnhancedWebSearcher:
    """Улучшенный поиск информации с интеллектуальным анализом результатов"""

    def __init__(self):
        self.cache_file = WEB_CACHE_FILE
        self.cache = FileManager.safe_load_json(self.cache_file, {})
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        """Инициализация сессии"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        """Закрытие сессии"""
        if self.session:
            await self.session.close()
            self.session = None

    def _save_cache(self):
        """Сохранение кеша"""
        FileManager.safe_save_json(self.cache_file, self.cache)

    def _clean_cache(self, max_age_hours: int = 24):
        """Очистка устаревшего кеша"""
        now = datetime.now()
        to_remove = []

        for key, cached in self.cache.items():
            try:
                cached_time = datetime.fromisoformat(cached['timestamp'])
                if now - cached_time > timedelta(hours=max_age_hours):
                    to_remove.append(key)
            except:
                to_remove.append(key)

        for key in to_remove:
            del self.cache[key]

        if to_remove:
            print(f"🧹 Очищено {len(to_remove)} устаревших записей кеша")
            self._save_cache()

    def _determine_cache_ttl(self, query: str) -> float:
        """Определение времени жизни кеша в зависимости от типа запроса"""
        query_lower = query.lower()

        # Очень быстро меняющиеся данные (5 минут)
        if any(word in query_lower for word in ['курс', 'цена биткоин', 'криптовалют', 'акци']):
            return 0.083  # 5 минут

        # Быстро меняющиеся данные (10 минут)
        if any(word in query_lower for word in ['погода', 'weather', 'temperature']):
            return 0.17  # 10 минут

        # Средне меняющиеся данные (30 минут)
        if any(word in query_lower for word in ['новости', 'news', 'breaking']):
            return 0.5  # 30 минут

        # Медленно меняющиеся данные (2 часа)
        if any(word in query_lower for word in ['цена', 'price', 'стоимость']):
            return 2.0

        # Статичные данные (6 часов)
        return 6.0

    async def search(self, query: str, max_results: int = 5, deep_fetch: bool = False) -> Dict[str, Any]:
        """
        Улучшенный поиск с опциональной глубокой загрузкой содержимого

        Args:
            query: Поисковый запрос
            max_results: Максимальное количество результатов
            deep_fetch: Загружать ли полное содержимое страниц

        Returns:
            Dict с результатами поиска и метаданными
        """
        # Проверка кеша
        cache_key = hashlib.md5(f"{query}_{deep_fetch}".encode()).hexdigest()
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            try:
                cached_time = datetime.fromisoformat(cached['timestamp'])
                cache_ttl = self._determine_cache_ttl(query)

                if datetime.now() - cached_time < timedelta(hours=cache_ttl):
                    print(f"📦 Использован кеш для: {query}")
                    return cached['data']
            except Exception as e:
                print(f"⚠️ Ошибка проверки кеша: {e}")

        if not DDGS_AVAILABLE:
            print("⚠️ DuckDuckGo Search недоступен")
            return {
                'success': False,
                'query': query,
                'results': [],
                'error': 'DuckDuckGo Search недоступен'
            }

        results = []
        search_metadata = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'deep_fetch': deep_fetch,
            'results_count': 0
        }

        try:
            if DDGS_ASYNC:
                results = await self._async_search(query, max_results)
            else:
                results = await self._sync_search(query, max_results)

            # Глубокая загрузка содержимого (если требуется)
            if deep_fetch and results:
                print(f"🔍 Глубокая загрузка содержимого для {len(results)} результатов...")
                results = await self._deep_fetch_content(results)

            search_metadata['results_count'] = len(results)

            if results:
                # Сохраняем в кеш
                result_data = {
                    'success': True,
                    'query': query,
                    'results': results,
                    'metadata': search_metadata
                }

                self.cache[cache_key] = {
                    'query': query,
                    'data': result_data,
                    'timestamp': datetime.now().isoformat()
                }
                self._save_cache()

                # Периодически чистим кеш
                if len(self.cache) > 100:
                    self._clean_cache()

                print(f"🔍 Найдено {len(results)} результатов для: {query}")

                return result_data
            else:
                print(f"⚠️ Нет результатов для: {query}")
                return {
                    'success': False,
                    'query': query,
                    'results': [],
                    'metadata': search_metadata,
                    'error': 'Результаты не найдены'
                }

        except Exception as e:
            print(f"⚠️ Ошибка веб-поиска: {type(e).__name__}: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'query': query,
                'results': [],
                'metadata': search_metadata,
                'error': str(e)
            }

    async def _async_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Асинхронный поиск через AsyncDDGS"""
        results = []
        try:
            async with AsyncDDGS() as ddgs:
                search_results = ddgs.text(query, max_results=max_results)

                if hasattr(search_results, '__aiter__'):
                    async for r in search_results:
                        results.append(self._format_result(r))
                else:
                    for r in search_results:
                        results.append(self._format_result(r))

        except Exception as e:
            print(f"⚠️ Ошибка async поиска: {e}")
            raise

        return results

    async def _sync_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Синхронный поиск через DDGS"""
        results = []
        try:
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: list(DDGS().text(query, max_results=max_results))
            )

            for r in search_results:
                results.append(self._format_result(r))

        except Exception as e:
            print(f"⚠️ Ошибка sync поиска: {e}")
            raise

        return results

    def _format_result(self, result: Dict) -> Dict[str, Any]:
        """Форматирование результата поиска"""
        return {
            'title': result.get('title', 'Без названия'),
            'url': result.get('href', result.get('link', '')),
            'snippet': result.get('body', result.get('snippet', 'Нет описания')),
            'source': 'duckduckgo',
            'content': None  # Будет заполнено при deep_fetch
        }

    async def _deep_fetch_content(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Глубокая загрузка содержимого страниц"""
        tasks = []
        for result in results[:3]:  # Ограничиваем 3 страницами для производительности
            tasks.append(self._fetch_and_attach_content(result))

        fetched_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Объединяем успешно загруженные результаты с остальными
        enhanced_results = []
        for i, result in enumerate(results):
            if i < len(fetched_results) and not isinstance(fetched_results[i], Exception):
                enhanced_results.append(fetched_results[i])
            else:
                enhanced_results.append(result)

        return enhanced_results

    async def _fetch_and_attach_content(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Загрузка и прикрепление содержимого к результату"""
        content = await self.fetch_page_content(result['url'], max_length=3000)
        result['content'] = content
        result['has_content'] = content is not None
        return result

    async def fetch_page_content(self, url: str, max_length: int = 5000) -> Optional[str]:
        """Загрузка содержимого страницы"""
        if not self.session:
            await self.initialize()

        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == 200:
                    content = await response.text()

                    if BEAUTIFULSOUP_AVAILABLE:
                        soup = BeautifulSoup(content, 'html.parser')
                        for script in soup(["script", "style"]):
                            script.decompose()
                        text = soup.get_text()
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        text = ' '.join(chunk for chunk in chunks if chunk)
                        return text[:max_length]
                    else:
                        text = re.sub(r'<[^>]+>', '', content)
                        text = re.sub(r'\s+', ' ', text).strip()
                        return text[:max_length]
                else:
                    print(f"⚠️ HTTP {response.status} для {url}")

        except asyncio.TimeoutError:
            print(f"⏱️ Таймаут при загрузке {url}")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки страницы: {e}")

        return None


# ==================== РАСШИРЕННАЯ СИСТЕМА ВООБРАЖЕНИЯ И СИНТЕЗА ====================
class CreativeSynthesisEngine:
    """
    🎨 ДВИЖОК ТВОРЧЕСКОГО СИНТЕЗА
    - Комбинирует информацию из разных источников
    - Создаёт новые идеи и концепции
    - Находит неожиданные связи
    - Генерирует инновационные решения
    """

    def __init__(self, memory_dir: str):
        self.memory_dir = memory_dir
        self.synthesis_file = os.path.join(memory_dir, "creative_synthesis.json")
        self.syntheses: List[Dict[str, Any]] = FileManager.safe_load_json(self.synthesis_file, [])

    async def synthesize_from_sources(
            self,
            sources: List[Dict[str, Any]],
            synthesis_type: str = 'creative'
    ) -> Optional[Dict[str, Any]]:
        """
        Синтез информации из множества источников

        Args:
            sources: Список источников информации (веб-результаты, память, факты)
            synthesis_type: Тип синтеза ('creative', 'analytical', 'comparative')
        """
        if len(sources) < 2:
            return None

        # Извлекаем ключевую информацию из каждого источника
        extracted_info = []
        all_keywords = set()

        for source in sources:
            info = self._extract_key_info(source)
            if info:
                extracted_info.append(info)
                all_keywords.update(info.get('keywords', []))

        if len(extracted_info) < 2:
            return None

        # Находим общие темы и уникальные аспекты
        common_themes = self._find_common_themes(extracted_info)
        unique_aspects = self._find_unique_aspects(extracted_info)

        # Создаём синтез
        synthesis = {
            'id': hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:12],
            'type': synthesis_type,
            'sources_count': len(sources),
            'common_themes': common_themes,
            'unique_aspects': unique_aspects,
            'all_keywords': list(all_keywords)[:15],
            'timestamp': datetime.now().isoformat(),
            'synthesis_text': self._generate_synthesis_text(
                extracted_info, common_themes, unique_aspects, synthesis_type
            )
        }

        self.syntheses.append(synthesis)
        self._save()

        return synthesis

    def _extract_key_info(self, source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Извлечение ключевой информации из источника"""
        text = ""

        # Определяем тип источника и извлекаем текст
        if 'content' in source and source['content']:
            text = source['content']
        elif 'snippet' in source:
            text = source['snippet']
        elif 'title' in source:
            text = source['title']
        else:
            return None

        keywords = MemoryUtils.extract_keywords(text, top_n=10)

        return {
            'text': text[:500],
            'keywords': keywords,
            'source_type': source.get('source', 'unknown'),
            'has_numbers': bool(re.search(r'\d+', text)),
            'length': len(text)
        }

    def _find_common_themes(self, extracted_info: List[Dict[str, Any]]) -> List[str]:
        """Поиск общих тем во всех источниках"""
        # Подсчитываем, сколько раз каждое ключевое слово встречается
        keyword_counts = Counter()

        for info in extracted_info:
            for keyword in info['keywords']:
                keyword_counts[keyword] += 1

        # Возвращаем слова, которые встречаются в нескольких источниках
        min_occurrence = max(2, len(extracted_info) // 2)
        common = [kw for kw, count in keyword_counts.items() if count >= min_occurrence]

        return common[:10]

    def _find_unique_aspects(self, extracted_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Поиск уникальных аспектов каждого источника"""
        all_keywords = set()
        for info in extracted_info:
            all_keywords.update(info['keywords'])

        unique_aspects = []

        for i, info in enumerate(extracted_info):
            # Находим ключевые слова, уникальные для этого источника
            other_keywords = set()
            for j, other_info in enumerate(extracted_info):
                if i != j:
                    other_keywords.update(other_info['keywords'])

            unique_to_source = set(info['keywords']) - other_keywords

            if unique_to_source:
                unique_aspects.append({
                    'source_index': i,
                    'unique_keywords': list(unique_to_source)[:5],
                    'text_preview': info['text'][:150]
                })

        return unique_aspects

    def _generate_synthesis_text(
            self,
            extracted_info: List[Dict[str, Any]],
            common_themes: List[str],
            unique_aspects: List[Dict[str, Any]],
            synthesis_type: str
    ) -> str:
        """Генерация текста синтеза"""
        if synthesis_type == 'creative':
            return self._generate_creative_synthesis(extracted_info, common_themes, unique_aspects)
        elif synthesis_type == 'analytical':
            return self._generate_analytical_synthesis(extracted_info, common_themes, unique_aspects)
        elif synthesis_type == 'comparative':
            return self._generate_comparative_synthesis(extracted_info, common_themes, unique_aspects)
        else:
            return "Синтез не удался"

    def _generate_creative_synthesis(
            self,
            extracted_info: List[Dict[str, Any]],
            common_themes: List[str],
            unique_aspects: List[Dict[str, Any]]
    ) -> str:
        """Творческий синтез с поиском неожиданных связей"""
        synthesis_parts = []

        if common_themes:
            synthesis_parts.append(
                f"🔗 Общие темы: {', '.join(common_themes[:5])}"
            )

        if len(unique_aspects) >= 2:
            # Создаём неожиданные связи между уникальными аспектами
            aspects_text = []
            for aspect in unique_aspects[:3]:
                keywords = aspect['unique_keywords'][:2]
                if keywords:
                    aspects_text.append(f"[{', '.join(keywords)}]")

            if len(aspects_text) >= 2:
                synthesis_parts.append(
                    f"💡 Неожиданные связи: {' ↔️ '.join(aspects_text)}"
                )

        synthesis_parts.append(
            f"✨ Синтезировано из {len(extracted_info)} источников информации"
        )

        return "\n".join(synthesis_parts)

    def _generate_analytical_synthesis(
            self,
            extracted_info: List[Dict[str, Any]],
            common_themes: List[str],
            unique_aspects: List[Dict[str, Any]]
    ) -> str:
        """Аналитический синтез с логическими связями"""
        synthesis_parts = []

        synthesis_parts.append(f"📊 Анализ {len(extracted_info)} источников:")

        if common_themes:
            synthesis_parts.append(f"  • Общие элементы: {', '.join(common_themes[:7])}")

        # Подсчёт источников с числами (данными)
        sources_with_data = sum(1 for info in extracted_info if info['has_numbers'])
        if sources_with_data > 0:
            synthesis_parts.append(f"  • Источников с данными: {sources_with_data}")

        if unique_aspects:
            synthesis_parts.append(f"  • Уникальных аспектов: {len(unique_aspects)}")

        return "\n".join(synthesis_parts)

    def _generate_comparative_synthesis(
            self,
            extracted_info: List[Dict[str, Any]],
            common_themes: List[str],
            unique_aspects: List[Dict[str, Any]]
    ) -> str:
        """Сравнительный синтез"""
        synthesis_parts = []

        synthesis_parts.append(f"⚖️ Сравнение {len(extracted_info)} источников:")

        if common_themes:
            synthesis_parts.append(f"  ✅ Согласованность: {', '.join(common_themes[:5])}")

        if unique_aspects:
            synthesis_parts.append(f"  ⚡ Различия:")
            for i, aspect in enumerate(unique_aspects[:3], 1):
                keywords = ', '.join(aspect['unique_keywords'][:3])
                synthesis_parts.append(f"     {i}. {keywords}")

        return "\n".join(synthesis_parts)

    async def cross_pollinate(
            self,
            concept1: str,
            concept2: str,
            associative_memory
    ) -> Optional[Dict[str, Any]]:
        """
        Кросс-опыление идей: создание новой концепции из двух несвязанных
        """
        # Получаем ассоциации для каждого концепта
        assoc1 = associative_memory.get_associations(concept1, min_strength=0.3, limit=10)
        assoc2 = associative_memory.get_associations(concept2, min_strength=0.3, limit=10)

        if not assoc1 or not assoc2:
            return None

        # Ищем мосты между концептами
        bridges = []
        for related1, strength1 in assoc1:
            for related2, strength2 in assoc2:
                combined_strength = (strength1 + strength2) / 2
                if related1 != related2:
                    bridges.append({
                        'bridge': [related1, related2],
                        'strength': combined_strength
                    })

        if not bridges:
            # Создаём прямое соединение
            new_concept = f"{concept1}+{concept2}"
            return {
                'type': 'direct_fusion',
                'concepts': [concept1, concept2],
                'new_concept': new_concept,
                'description': f"Прямое слияние концептов: {concept1} и {concept2}"
            }

        # Выбираем самый сильный мост
        best_bridge = max(bridges, key=lambda x: x['strength'])

        return {
            'type': 'bridged_fusion',
            'concepts': [concept1, concept2],
            'bridge': best_bridge['bridge'],
            'strength': best_bridge['strength'],
            'description': (
                f"Связь между {concept1} и {concept2} "
                f"через {' → '.join(best_bridge['bridge'])}"
            )
        }

    def _save(self):
        """Сохранение синтезов"""
        # Ограничиваем историю
        if len(self.syntheses) > 100:
            self.syntheses = self.syntheses[-100:]

        FileManager.safe_save_json(self.synthesis_file, self.syntheses)

    def get_recent_syntheses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Получение недавних синтезов"""
        return self.syntheses[-limit:]


# ==================== АССОЦИАТИВНАЯ ПАМЯТЬ ====================
class AssociativeMemory:
    """Ассоциативная память с улучшенными возможностями"""

    def __init__(self, memory_dir: str):
        self.memory_dir = memory_dir
        self.associations_file = os.path.join(memory_dir, "associations.json")
        self.imagination_file = os.path.join(memory_dir, "imagination.json")

        self.associations: Dict[str, Dict[str, float]] = FileManager.safe_load_json(
            self.associations_file, {}
        )

        self.imaginations: List[Dict[str, Any]] = FileManager.safe_load_json(
            self.imagination_file, []
        )

    def add_association(self, concept1: str, concept2: str, strength: float = 0.5):
        """Добавление ассоциации между концептами"""
        concept1 = concept1.lower().strip()
        concept2 = concept2.lower().strip()

        if concept1 == concept2:
            return

        if concept1 not in self.associations:
            self.associations[concept1] = {}
        if concept2 not in self.associations:
            self.associations[concept2] = {}

        current_strength = self.associations[concept1].get(concept2, 0.0)
        self.associations[concept1][concept2] = min(1.0, current_strength + strength)

        current_strength = self.associations[concept2].get(concept1, 0.0)
        self.associations[concept2][concept1] = min(1.0, current_strength + strength)

    def get_associations(self, concept: str, min_strength: float = 0.3, limit: int = 10) -> List[Tuple[str, float]]:
        """Получение ассоциаций для концепта"""
        concept = concept.lower().strip()

        if concept not in self.associations:
            return []

        associations = [
            (related, strength)
            for related, strength in self.associations[concept].items()
            if strength >= min_strength
        ]

        associations.sort(key=lambda x: x[1], reverse=True)
        return associations[:limit]

    def find_path(self, start: str, end: str, max_depth: int = 3) -> Optional[List[str]]:
        """Поиск ассоциативного пути между концептами (BFS)"""
        start = start.lower().strip()
        end = end.lower().strip()

        if start == end:
            return [start]

        if start not in self.associations or end not in self.associations:
            return None

        queue = [(start, [start])]
        visited = {start}

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            for neighbor, strength in self.associations.get(current, {}).items():
                if neighbor in visited or strength < 0.3:
                    continue

                new_path = path + [neighbor]

                if neighbor == end:
                    return new_path

                visited.add(neighbor)
                queue.append((neighbor, new_path))

        return None

    def imagine(self, base_concepts: List[str]) -> Optional[Dict[str, Any]]:
        """Воображение: синтез нового концепта из базовых"""
        if len(base_concepts) < 2:
            return None

        all_associations = defaultdict(float)

        for concept in base_concepts:
            associations = self.get_associations(concept, min_strength=0.4)
            for related, strength in associations:
                all_associations[related] += strength

        if not all_associations:
            return None

        common = max(all_associations.items(), key=lambda x: x[1])

        imagination = {
            'id': hashlib.md5(f"{'_'.join(base_concepts)}_{common[0]}".encode()).hexdigest()[:12],
            'base_concepts': base_concepts,
            'imagined_concept': common[0],
            'strength': common[1] / len(base_concepts),
            'timestamp': datetime.now().isoformat(),
            'description': f"Синтез из {', '.join(base_concepts)} через {common[0]}"
        }

        self.imaginations.append(imagination)
        return imagination

    def save(self):
        """Сохранение ассоциаций и воображения"""
        FileManager.safe_save_json(self.associations_file, self.associations)
        FileManager.safe_save_json(self.imagination_file, self.imaginations)


# ==================== УЛУЧШЕННАЯ СИСТЕМА ПАМЯТИ ====================
class EnhancedMemorySystem:
    """Улучшенная система памяти"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory_dir = os.path.join(MEMORY_DIR, f"user_{user_id}")
        os.makedirs(self.memory_dir, exist_ok=True)

        self.short_term_file = os.path.join(self.memory_dir, "short_term.json")
        self.long_term_file = os.path.join(self.memory_dir, "long_term.json")
        self.working_file = os.path.join(self.memory_dir, "working.json")
        self.episodic_file = os.path.join(self.memory_dir, "episodic.json")
        self.patterns_file = os.path.join(self.memory_dir, "patterns.json")
        self.preferences_file = os.path.join(self.memory_dir, "preferences.json")
        self.metadata_file = os.path.join(self.memory_dir, "metadata.json")

        self.short_term: List[Dict] = FileManager.safe_load_json(self.short_term_file, [])
        self.long_term: List[Dict] = FileManager.safe_load_json(self.long_term_file, [])
        self.working_memory: Dict = FileManager.safe_load_json(self.working_file, {
            'current_topic': None,
            'conversation_context': [],
            'active_goals': [],
            'timestamp': datetime.now().isoformat()
        })
        self.episodic_memory: List[Dict] = FileManager.safe_load_json(self.episodic_file, [])

        self.patterns: Dict = FileManager.safe_load_json(self.patterns_file, {
            'communication_style': 'neutral',
            'frequent_topics': [],
            'emotional_markers': {},
            'time_preferences': {},
            'interaction_patterns': {},
            'cognitive_patterns': {},
            'learning_speed': 1.0,
            'last_pattern_update': datetime.now().isoformat()
        })

        self.preferences: Dict = FileManager.safe_load_json(self.preferences_file, {
            'explicit': {},
            'inferred': {},
            'confirmed': {}
        })

        # Загружаем метаданные с безопасной инициализацией
        self.metadata: Dict = FileManager.safe_load_json(self.metadata_file, {})

        # Обеспечиваем наличие всех необходимых ключей
        default_metadata = {
            'total_messages': 0,
            'facts_extracted': 0,
            'patterns_identified': 0,
            'auto_transfers': 0,
            'associations_created': 0,
            'imaginations_generated': 0,
            'syntheses_created': 0,
            'first_interaction': datetime.now().isoformat(),
            'last_interaction': datetime.now().isoformat(),
            'session_count': 0,
            'learning_cycles': 0
        }

        # Добавляем отсутствующие ключи
        for key, value in default_metadata.items():
            if key not in self.metadata:
                self.metadata[key] = value

        self.associative = AssociativeMemory(self.memory_dir)
        self.creative_synthesis = CreativeSynthesisEngine(self.memory_dir)

        self._changes_count = 0
        self._autosave_threshold = 3

        print(f"🧠 Память загружена для {user_id}:")
        print(f"   ST={len(self.short_term)}, LT={len(self.long_term)}, "
              f"Episodes={len(self.episodic_memory)}")
        print(f"   Ассоциации={len(self.associative.associations)}, "
              f"Воображения={len(self.associative.imaginations)}")
        print(f"   Синтезы={len(self.creative_synthesis.syntheses)}")

    async def analyze_and_store_message(
            self,
            role: str,
            content: str,
            llm_caller: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Глубокий анализ сообщения"""
        timestamp = datetime.now()

        entry = {
            'role': role,
            'content': content,
            'timestamp': timestamp.isoformat(),
            'word_count': len(content.split()),
            'has_question': '?' in content,
            'has_numbers': bool(re.search(r'\d', content)),
            'keywords': MemoryUtils.extract_keywords(content),
            'extracted_facts': [],
            'detected_emotions': [],
            'topic_tags': [],
            'concepts': []
        }

        quick_analysis = self._quick_analysis(content, role)
        entry.update(quick_analysis)

        if llm_caller and role == 'user' and len(content) > 10:
            try:
                deep_analysis = await self._deep_analysis(content, llm_caller)

                entry['extracted_facts'].extend(deep_analysis.get('facts', []))
                entry['detected_emotions'].extend(deep_analysis.get('emotions', []))
                entry['topic_tags'].extend(deep_analysis.get('topics', []))
                entry['concepts'].extend(deep_analysis.get('concepts', []))

                for fact in deep_analysis.get('important_facts', []):
                    await self.add_to_long_term(
                        fact['content'],
                        category=fact.get('category', 'general'),
                        importance=fact.get('importance', 0.7),
                        source='auto_extraction'
                    )

                concepts = entry['concepts']
                for i, concept1 in enumerate(concepts):
                    for concept2 in concepts[i + 1:]:
                        self.associative.add_association(concept1, concept2, strength=0.4)
                        self.metadata['associations_created'] += 1
            except Exception as e:
                print(f"⚠️ Ошибка глубокого анализа: {e}")

        self.short_term.append(entry)
        self._update_working_memory(entry)

        if self._should_create_episode():
            await self._create_episode()

        if len(self.short_term) > 50:
            await self._intelligent_transfer_to_long_term()
            self.short_term = self.short_term[-30:]

        await self._update_patterns(entry)
        await self._learn_from_interaction(entry)

        self.metadata['total_messages'] += 1
        self.metadata['last_interaction'] = timestamp.isoformat()

        await self._auto_save()

        return entry

    def _quick_analysis(self, content: str, role: str) -> Dict[str, Any]:
        """Быстрый анализ без LLM"""
        result = {
            'sentiment': 'neutral',
            'urgency': 'normal',
            'length_category': 'medium'
        }

        word_count = len(content.split())
        if word_count < 10:
            result['length_category'] = 'short'
        elif word_count > 50:
            result['length_category'] = 'long'

        positive_words = ['хорошо', 'отлично', 'супер', 'класс', 'спасибо', 'thanks', 'great', 'good', 'прекрасно']
        negative_words = ['плохо', 'ужасно', 'не', 'нет', 'bad', 'terrible', 'wrong', 'ошибка', 'проблема']

        content_lower = content.lower()
        positive_count = sum(1 for w in positive_words if w in content_lower)
        negative_count = sum(1 for w in negative_words if w in content_lower)

        if positive_count > negative_count:
            result['sentiment'] = 'positive'
        elif negative_count > positive_count:
            result['sentiment'] = 'negative'

        urgent_markers = ['срочно', 'быстро', 'важно', 'urgent', 'asap', '!!!', 'скорее']
        if any(marker in content_lower for marker in urgent_markers):
            result['urgency'] = 'high'

        return result

    async def _deep_analysis(self, content: str, llm_caller: callable) -> Dict[str, Any]:
        """Глубокий анализ через LLM"""
        prompt = f"""Проанализируй следующее сообщение и извлеки:
1. Факты (конкретная информация)
2. Эмоции (sentiment)
3. Темы (категории)
4. Концепты (ключевые идеи)
5. Важность (0.0-1.0)

Сообщение: "{content}"

Ответь в JSON формате:
{{
    "facts": ["факт1", "факт2"],
    "emotions": ["эмоция1"],
    "topics": ["тема1"],
    "concepts": ["концепт1", "концепт2"],
    "important_facts": [
        {{"content": "важный факт", "category": "категория", "importance": 0.8}}
    ]
}}"""

        try:
            response = await llm_caller(prompt, temperature=0.3, max_tokens=500)

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"⚠️ Ошибка глубокого анализа: {e}")

        return {
            'facts': [],
            'emotions': [],
            'topics': [],
            'concepts': [],
            'important_facts': []
        }

    def _update_working_memory(self, entry: Dict[str, Any]):
        """Обновление рабочей памяти"""
        context_entry = {
            'role': entry['role'],
            'content': entry['content'][:200],
            'timestamp': entry['timestamp'],
            'keywords': entry.get('keywords', [])
        }

        self.working_memory['conversation_context'].append(context_entry)

        if len(self.working_memory['conversation_context']) > 10:
            self.working_memory['conversation_context'] = \
                self.working_memory['conversation_context'][-10:]

        if entry.get('topic_tags'):
            self.working_memory['current_topic'] = entry['topic_tags'][0]

        self.working_memory['timestamp'] = datetime.now().isoformat()

    def _should_create_episode(self) -> bool:
        """Проверка необходимости создания эпизода"""
        if len(self.short_term) >= 20:
            return True

        if len(self.short_term) >= 5:
            recent_topics = set()
            for msg in self.short_term[-5:]:
                recent_topics.update(msg.get('topic_tags', []))

            if len(recent_topics) > 2:
                return True

        return False

    async def _create_episode(self):
        """Создание эпизодической памяти"""
        if len(self.short_term) < 3:
            return

        messages = self.short_term[-20:]
        summary = self._summarize_episode(messages)

        episode = {
            'id': hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:12],
            'start_time': messages[0]['timestamp'],
            'end_time': messages[-1]['timestamp'],
            'message_count': len(messages),
            'summary': summary,
            'main_topics': self._extract_main_topics(messages),
            'emotional_tone': self._determine_emotional_tone(messages),
            'importance': self._calculate_episode_importance(messages),
            'concepts': self._extract_episode_concepts(messages)
        }

        self.episodic_memory.append(episode)

        if len(self.episodic_memory) > 50:
            self.episodic_memory = self.episodic_memory[-50:]

        print(f"📖 Создан эпизод: {summary[:50]}...")

    def _summarize_episode(self, messages: List[Dict]) -> str:
        """Суммаризация эпизода"""
        topics = []
        for msg in messages:
            topics.extend(msg.get('topic_tags', []))

        topic_counts = Counter(topics)
        main_topics = [topic for topic, _ in topic_counts.most_common(3)]

        if main_topics:
            return f"Обсуждение: {', '.join(main_topics)}"
        else:
            return f"Разговор из {len(messages)} сообщений"

    def _extract_main_topics(self, messages: List[Dict]) -> List[str]:
        """Извлечение главных тем"""
        topics = []
        for msg in messages:
            topics.extend(msg.get('topic_tags', []))

        topic_counts = Counter(topics)
        return [topic for topic, _ in topic_counts.most_common(5)]

    def _determine_emotional_tone(self, messages: List[Dict]) -> str:
        """Определение эмоционального тона эпизода"""
        sentiments = [msg.get('sentiment', 'neutral') for msg in messages]
        sentiment_counts = Counter(sentiments)

        most_common = sentiment_counts.most_common(1)
        return most_common[0][0] if most_common else 'neutral'

    def _calculate_episode_importance(self, messages: List[Dict]) -> float:
        """Вычисление важности эпизода"""
        total_importance = 0.0

        for msg in messages:
            msg_importance = MemoryUtils.calculate_importance(msg)
            total_importance += msg_importance

        return total_importance / len(messages) if messages else 0.5

    def _extract_episode_concepts(self, messages: List[Dict]) -> List[str]:
        """Извлечение концептов из эпизода"""
        concepts = set()
        for msg in messages:
            concepts.update(msg.get('concepts', []))

        return list(concepts)[:10]

    async def _intelligent_transfer_to_long_term(self):
        """Умный перенос в долговременную память"""
        if len(self.short_term) < 10:
            return

        transferred = 0

        for entry in self.short_term:
            importance = MemoryUtils.calculate_importance(entry)

            should_transfer = False

            if importance >= 0.7:
                should_transfer = True

            if len(entry.get('extracted_facts', [])) > 0:
                should_transfer = True

            if entry.get('detected_emotions') and importance >= 0.6:
                should_transfer = True

            concepts = entry.get('concepts', [])
            if concepts:
                concept_frequency = sum(
                    1 for msg in self.short_term
                    if any(c in msg.get('concepts', []) for c in concepts)
                )
                if concept_frequency >= 3:
                    should_transfer = True

            if should_transfer:
                await self.add_to_long_term(
                    entry['content'],
                    category=entry.get('topic_tags', ['general'])[0] if entry.get('topic_tags') else 'general',
                    importance=importance,
                    source='intelligent_transfer',
                    metadata={
                        'keywords': entry.get('keywords', []),
                        'concepts': entry.get('concepts', []),
                        'emotions': entry.get('detected_emotions', []),
                        'original_timestamp': entry['timestamp']
                    }
                )
                transferred += 1
                self.metadata['auto_transfers'] += 1

        if transferred > 0:
            print(f"🔄 Интеллектуальный перенос: {transferred} записей в LT память")

    async def add_to_long_term(
            self,
            content: str,
            category: str = 'general',
            importance: float = 0.5,
            source: str = 'manual',
            metadata: Optional[Dict] = None
    ):
        """Добавление в долгосрочную память"""
        for existing in self.long_term:
            similarity = MemoryUtils.calculate_similarity(content, existing['content'])
            if similarity > 0.8:
                existing['importance'] = min(1.0, existing['importance'] + 0.1)
                existing['access_count'] += 1
                existing['last_accessed'] = datetime.now().isoformat()
                return

        entry = {
            'id': hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:12],
            'content': content,
            'category': category,
            'importance': importance,
            'source': source,
            'created': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'access_count': 0,
            'keywords': MemoryUtils.extract_keywords(content),
            'metadata': metadata or {}
        }

        self.long_term.append(entry)
        self.metadata['facts_extracted'] += 1

        keywords = entry['keywords']
        for i, kw1 in enumerate(keywords):
            for kw2 in keywords[i + 1:]:
                self.associative.add_association(kw1, kw2, strength=0.3)

        if len(self.long_term) > 500:
            self.long_term.sort(
                key=lambda x: (x['importance'], x['access_count']),
                reverse=True
            )
            self.long_term = self.long_term[:400]

    async def _update_patterns(self, entry: Dict[str, Any]):
        """Обновление паттернов поведения"""
        if entry['role'] != 'user':
            return

        word_count = entry['word_count']
        if word_count < 10:
            style = 'краткий'
        elif word_count < 30:
            style = 'средний'
        else:
            style = 'подробный'

        self.patterns['communication_style'] = style

        topics = entry.get('topic_tags', [])
        for topic in topics:
            if topic not in self.patterns['frequent_topics']:
                self.patterns['frequent_topics'].append(topic)

        if len(self.patterns['frequent_topics']) > 20:
            self.patterns['frequent_topics'] = self.patterns['frequent_topics'][-20:]

        emotions = entry.get('detected_emotions', [])
        for emotion in emotions:
            if emotion not in self.patterns['emotional_markers']:
                self.patterns['emotional_markers'][emotion] = 0
            self.patterns['emotional_markers'][emotion] += 1

        await self._identify_cognitive_patterns(entry)

        self.patterns['last_pattern_update'] = datetime.now().isoformat()
        self.metadata['patterns_identified'] += 1

    async def _identify_cognitive_patterns(self, entry: Dict[str, Any]):
        """Выявление когнитивных паттернов мышления"""
        content = entry['content'].lower()

        patterns = {
            'analytical': ['потому что', 'следовательно', 'анализ', 'данные', 'факты'],
            'creative': ['идея', 'представь', 'возможно', 'креативно', 'воображение'],
            'practical': ['сделать', 'как', 'практически', 'применить', 'использовать'],
            'emotional': ['чувствую', 'переживаю', 'радость', 'грусть', 'волнуюсь'],
            'questioning': ['почему', 'как', 'что', 'зачем', 'когда']
        }

        if 'cognitive_patterns' not in self.patterns:
            self.patterns['cognitive_patterns'] = {}

        for pattern_type, markers in patterns.items():
            count = sum(1 for marker in markers if marker in content)
            if count > 0:
                if pattern_type not in self.patterns['cognitive_patterns']:
                    self.patterns['cognitive_patterns'][pattern_type] = 0
                self.patterns['cognitive_patterns'][pattern_type] += count

    async def _learn_from_interaction(self, entry: Dict[str, Any]):
        """Обучение на основе взаимодействия"""
        self.metadata['learning_cycles'] += 1

        total_messages = self.metadata['total_messages']
        if total_messages > 100:
            self.patterns['learning_speed'] = max(0.5, 1.0 - (total_messages / 10000))

    async def search_memory(
            self,
            query: str,
            search_in: str = 'all',
            limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Поиск в памяти"""
        results = []
        query_lower = query.lower()

        if search_in in ['all', 'long_term']:
            for mem in self.long_term:
                if query_lower in mem['content'].lower():
                    score = 1.0
                elif any(kw in query_lower for kw in mem.get('keywords', [])):
                    score = 0.8
                else:
                    score = MemoryUtils.calculate_similarity(query, mem['content'])

                if score > 0.3:
                    results.append({
                        'source': 'long_term',
                        'score': score,
                        'content': mem['content'],
                        'category': mem.get('category', 'general'),
                        'importance': mem.get('importance', 0.5),
                        'metadata': mem.get('metadata', {})
                    })

        if search_in in ['all', 'episodic']:
            for episode in self.episodic_memory:
                if query_lower in episode['summary'].lower():
                    results.append({
                        'source': 'episodic',
                        'score': 0.9,
                        'content': episode['summary'],
                        'topics': episode.get('main_topics', []),
                        'importance': episode.get('importance', 0.5)
                    })

        keywords = MemoryUtils.extract_keywords(query)
        for keyword in keywords:
            associations = self.associative.get_associations(keyword, min_strength=0.4, limit=5)
            for related, strength in associations:
                results.append({
                    'source': 'associative',
                    'score': strength,
                    'content': f"Ассоциация: {keyword} → {related}",
                    'related_concept': related,
                    'strength': strength
                })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

    async def imagine_from_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Воображение на основе запроса"""
        keywords = MemoryUtils.extract_keywords(query, top_n=3)

        if len(keywords) < 2:
            return None

        imagination = self.associative.imagine(keywords)

        if imagination:
            self.metadata['imaginations_generated'] += 1
            print(f"💭 Воображение: {imagination['description']}")

        return imagination

    def get_context_for_llm(self, max_tokens: int = 2000) -> str:
        """Формирование контекста для LLM"""
        context_parts = []

        if self.working_memory.get('conversation_context'):
            context_parts.append("=== ТЕКУЩИЙ РАЗГОВОР ===")
            for msg in self.working_memory['conversation_context'][-5:]:
                context_parts.append(f"{msg['role']}: {msg['content']}")

        if self.long_term:
            relevant_facts = sorted(
                self.long_term,
                key=lambda x: (x['importance'], x['access_count']),
                reverse=True
            )[:10]

            if relevant_facts:
                context_parts.append("\n=== ИЗВЕСТНЫЕ ФАКТЫ ===")
                for fact in relevant_facts:
                    context_parts.append(f"• {fact['content']}")

        context_parts.append(f"\n=== ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ ===")
        context_parts.append(f"Стиль: {self.patterns.get('communication_style', 'neutral')}")

        if self.patterns.get('frequent_topics'):
            topics = ', '.join(self.patterns['frequent_topics'][:5])
            context_parts.append(f"Интересы: {topics}")

        if self.episodic_memory:
            recent_episodes = self.episodic_memory[-3:]
            context_parts.append("\n=== НЕДАВНИЕ РАЗГОВОРЫ ===")
            for ep in recent_episodes:
                context_parts.append(f"• {ep['summary']}")

        context = "\n".join(context_parts)

        if len(context) > max_tokens * 4:
            context = context[:max_tokens * 4]

        return context

    async def _auto_save(self):
        """Автосохранение"""
        self._changes_count += 1

        if self._changes_count >= self._autosave_threshold:
            self._save_all()
            self._changes_count = 0

    def _save_all(self):
        """Сохранение всех данных"""
        FileManager.safe_save_json(self.short_term_file, self.short_term)
        FileManager.safe_save_json(self.long_term_file, self.long_term)
        FileManager.safe_save_json(self.working_file, self.working_memory)
        FileManager.safe_save_json(self.episodic_file, self.episodic_memory)
        FileManager.safe_save_json(self.patterns_file, self.patterns)
        FileManager.safe_save_json(self.preferences_file, self.preferences)
        FileManager.safe_save_json(self.metadata_file, self.metadata)

        self.associative.save()

    def get_stats(self) -> Dict[str, Any]:
        """Статистика памяти"""
        return {
            'short_term_count': len(self.short_term),
            'long_term_count': len(self.long_term),
            'episodic_count': len(self.episodic_memory),
            'total_messages': self.metadata['total_messages'],
            'facts_extracted': self.metadata['facts_extracted'],
            'patterns_identified': self.metadata['patterns_identified'],
            'auto_transfers': self.metadata['auto_transfers'],
            'associations': len(self.associative.associations),
            'imaginations': len(self.associative.imaginations),
            'syntheses': self.metadata.get('syntheses_created', 0),
            'learning_cycles': self.metadata['learning_cycles'],
            'communication_style': self.patterns['communication_style'],
            'frequent_topics': self.patterns['frequent_topics'][:10],
            'cognitive_patterns': self.patterns.get('cognitive_patterns', {}),
            'learning_speed': self.patterns.get('learning_speed', 1.0)
        }


# ==================== LLM ИНТЕРФЕЙС ====================
class LLMInterface:
    """Интерфейс для работы с LM Studio"""

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        """Инициализация сессии"""
        timeout = aiohttp.ClientTimeout(total=60, connect=10)
        self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        """Закрытие сессии"""
        if self.session:
            await self.session.close()

    async def generate(
            self,
            prompt: str,
            system: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 2000
    ) -> str:
        """Генерация ответа"""
        if not self.session:
            await self.initialize()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            async with self.session.post(
                    self.api_url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error_text = await response.text()
                    print(f"⚠️ LLM ошибка {response.status}: {error_text[:200]}")
                    return "Не удалось получить ответ от LLM"
        except asyncio.TimeoutError:
            print(f"⏱️ Таймаут LLM запроса")
            return "Превышено время ожидания ответа от LLM"
        except Exception as e:
            print(f"⚠️ LLM исключение: {e}")
            return f"Ошибка LLM: {str(e)}"


# ==================== УЛУЧШЕННЫЙ МЕНЕДЖЕР ОБРАБОТКИ ЗАПРОСОВ ====================
class QueryManager:
    """Управление обработкой запросов с улучшенным веб-поиском и синтезом"""

    def __init__(self, user_id: str, llm: LLMInterface, web_searcher: EnhancedWebSearcher):
        self.user_id = user_id
        self.llm = llm
        self.web_searcher = web_searcher
        self.memory = EnhancedMemorySystem(user_id)

    async def initialize(self):
        """Инициализация"""
        await self.web_searcher.initialize()
        print(f"🎯 QueryManager инициализирован для user_{self.user_id}")

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Обработка запроса пользователя"""
        # 1. Анализируем и сохраняем запрос
        await self.memory.analyze_and_store_message(
            role='user',
            content=query,
            llm_caller=self.llm.generate
        )

        # 2. Определяем стратегию ответа
        strategy = await self._determine_strategy(query)
        print(f"📊 Стратегия для '{query[:50]}...': {strategy}")

        # 3. Обрабатываем в зависимости от стратегии
        result = {'strategy': strategy, 'confidence': 0.0, 'answer': ''}

        if strategy == 'time_query':
            result = await self._handle_time_query(query)
        elif strategy == 'direct_answer':
            result = await self._handle_direct_answer(query)
        elif strategy == 'web_search':
            result = await self._handle_web_search(query)
        elif strategy == 'web_search_deep':
            result = await self._handle_web_search_deep(query)
        elif strategy == 'memory_search':
            result = await self._handle_memory_search(query)
        elif strategy == 'imagination':
            result = await self._handle_imagination(query)
        elif strategy == 'synthesis':
            result = await self._handle_synthesis(query)
        else:
            result = await self._handle_llm_generation(query)

        # 4. Сохраняем ответ
        await self.memory.analyze_and_store_message(
            role='assistant',
            content=result['answer']
        )

        return result

    async def _determine_strategy(self, query: str) -> str:
        """Определение стратегии обработки с приоритетами"""
        query_lower = query.lower()

        # ПРИОРИТЕТ 1: Прямой ответ (вычисления)
        if re.search(r'\d+\s*[\+\-\*/]\s*\d+', query):
            return 'direct_answer'

        # ПРИОРИТЕТ 2: Глубокий веб-поиск (анализ нескольких источников)
        deep_web_triggers = [
            'сравни', 'анализ', 'обзор', 'что лучше', 'различия',
            'compare', 'analysis', 'review', 'differences', 'vs'
        ]
        if any(trigger in query_lower for trigger in deep_web_triggers):
            return 'web_search_deep'

        # ПРИОРИТЕТ 3: Обычный веб-поиск
        web_triggers = [
            'курс', 'погода', 'новости', 'цена', 'купить',
            'найди', 'поищи', 'что такое', 'кто такой',
            'стоимость', 'актуальн', 'последн', 'свежи',
            'search', 'find', 'weather', 'news', 'price'
        ]
        if any(trigger in query_lower for trigger in web_triggers):
            pure_time_triggers = ['который час', 'сколько времени', 'какое время']
            if not any(trigger in query_lower for trigger in pure_time_triggers):
                return 'web_search'

        # ПРИОРИТЕТ 4: Синтез информации
        synthesis_triggers = [
            'синтезируй', 'объедини', 'соедини', 'скомбинируй',
            'synthesize', 'combine', 'merge', 'integrate'
        ]
        if any(trigger in query_lower for trigger in synthesis_triggers):
            return 'synthesis'

        # ПРИОРИТЕТ 5: Запросы времени и даты
        time_triggers = [
            'который час', 'сколько времени', 'какое время',
            'какая дата', 'какое число', 'какой день недели'
        ]
        if any(trigger in query_lower for trigger in time_triggers):
            return 'time_query'

        general_time_words = ['время', 'дата', 'день', 'вчера', 'завтра']
        if any(word in query_lower for word in general_time_words):
            significant_words = len([w for w in query_lower.split() if len(w) > 3])
            if significant_words <= 2:
                return 'time_query'

        # ПРИОРИТЕТ 6: Поиск в памяти
        memory_triggers = ['помнишь', 'ты знаешь', 'что я говорил', 'расскажи обо мне', 'наши разговоры']
        if any(trigger in query_lower for trigger in memory_triggers):
            return 'memory_search'

        # ПРИОРИТЕТ 7: Воображение
        imagination_triggers = ['представь', 'вообрази', 'придумай', 'создай идею']
        if any(trigger in query_lower for trigger in imagination_triggers):
            return 'imagination'

        # ПРИОРИТЕТ 8: LLM генерация (по умолчанию)
        return 'llm_generation'

    async def _handle_time_query(self, query: str) -> Dict[str, Any]:
        """Обработка запросов о времени и дате"""
        time_info = TimeUtils.get_current_time_info()
        query_lower = query.lower()

        if 'время' in query_lower or 'час' in query_lower:
            answer = f"⏰ Сейчас: **{time_info['formatted']}**"
        elif 'дата' in query_lower or 'число' in query_lower:
            answer = f"📅 Сегодня: **{time_info['datetime'].strftime('%d.%m.%Y')}** ({time_info['weekday_name']})"
        elif 'день' in query_lower:
            answer = f"📆 Сегодня: **{time_info['weekday_name']}**, {time_info['datetime'].strftime('%d.%m.%Y')}"
        else:
            time_of_day = TimeUtils.get_time_of_day()
            answer = (
                f"⏰ **Текущее время:** {time_info['datetime'].strftime('%H:%M:%S')}\n"
                f"📅 **Дата:** {time_info['datetime'].strftime('%d.%m.%Y')}\n"
                f"📆 **День недели:** {time_info['weekday_name']}\n"
                f"🌅 **Время суток:** {time_of_day}"
            )

        relative_time = TimeUtils.parse_relative_time(query)
        if relative_time and relative_time != time_info['datetime']:
            answer += f"\n\n🔄 **{query}:** {relative_time.strftime('%d.%m.%Y %H:%M')}"

        return {
            'strategy': 'time_query',
            'confidence': 1.0,
            'answer': answer
        }

    async def _handle_direct_answer(self, query: str) -> Dict[str, Any]:
        """Прямой ответ (вычисления)"""
        math_match = re.search(r'(\d+\.?\d*)\s*([\+\-\*/])\s*(\d+\.?\d*)', query)
        if math_match:
            try:
                a = float(math_match.group(1))
                op = math_match.group(2)
                b = float(math_match.group(3))

                operations = {
                    '+': a + b,
                    '-': a - b,
                    '*': a * b,
                    '/': a / b if b != 0 else None
                }

                result = operations.get(op)
                if result is None:
                    return {
                        'strategy': 'direct_answer',
                        'confidence': 0.5,
                        'answer': '❌ Деление на ноль невозможно'
                    }

                if isinstance(result, float) and result.is_integer():
                    result = int(result)

                return {
                    'strategy': 'direct_answer',
                    'confidence': 1.0,
                    'answer': f"🔢 **Результат:** `{a} {op} {b} = {result}`"
                }
            except Exception as e:
                print(f"⚠️ Ошибка вычисления: {e}")

        return {
            'strategy': 'direct_answer',
            'confidence': 0.3,
            'answer': 'Не удалось выполнить вычисление'
        }

    async def _handle_web_search(self, query: str) -> Dict[str, Any]:
        """Обычный веб-поиск с ТРЁХУРОВНЕВОЙ системой попыток"""
        try:
            # ПОПЫТКА 1: Оптимизированный запрос
            search_query = self._optimize_search_query(query)
            print(f"🔍 Попытка 1: '{search_query}'")

            search_result = await self.web_searcher.search(search_query, max_results=10, deep_fetch=False)

            if not search_result['success'] or not search_result['results']:
                return {
                    'strategy': 'web_search',
                    'confidence': 0.0,
                    'answer': f'❌ Не удалось найти информацию по запросу: "{query}"\n\nПопробуйте переформулировать запрос.'
                }

            results = search_result['results']

            # ФИЛЬТРАЦИЯ результатов
            filtered_results = self._filter_relevant_results(results, query, search_query)

            # ПОПЫТКА 2: Упрощенный запрос (если результатов нет)
            if not filtered_results:
                print(f"⚠️ Попытка 2: упрощенный запрос")
                simple_query = self._create_simple_query(query)
                print(f"🔍 Попытка 2: '{simple_query}'")

                search_result = await self.web_searcher.search(simple_query, max_results=10, deep_fetch=False)

                if search_result['success'] and search_result['results']:
                    filtered_results = self._filter_relevant_results(search_result['results'], query, simple_query)

            # ПОПЫТКА 3: Английский запрос (для курсов валют)
            if not filtered_results and 'курс' in query.lower():
                print(f"⚠️ Попытка 3: английский запрос")
                english_query = self._create_english_currency_query(query)
                print(f"🔍 Попытка 3: '{english_query}'")

                search_result = await self.web_searcher.search(english_query, max_results=10, deep_fetch=False)

                if search_result['success'] and search_result['results']:
                    filtered_results = self._filter_relevant_results(search_result['results'], query, english_query)

            # Если все 3 попытки не дали результатов
            if not filtered_results:
                return {
                    'strategy': 'web_search',
                    'confidence': 0.2,
                    'answer': (
                        f'⚠️ Не удалось найти релевантную информацию.\n\n'
                        f'**Попробовано:**\n'
                        f'1️⃣ Оптимизированный поиск\n'
                        f'2️⃣ Упрощенный поиск\n'
                        f'3️⃣ Альтернативный поиск\n\n'
                        f'💡 **Рекомендации:**\n'
                        f'• Уточните запрос\n'
                        f'• Используйте конкретные термины\n'
                        f'• Попробуйте другую формулировку'
                    )
                }

            # Формируем ответ из отфильтрованных результатов
            answer = f"🔍 **Результаты поиска:** `{query}`\n\n"

            for i, res in enumerate(filtered_results[:5], 1):
                title = res['title'][:100]
                snippet = res['snippet'][:300]
                url = res['url']
                relevance = res.get('relevance_score', 0.5)

                answer += f"**{i}. {title}**\n"
                answer += f"{snippet}...\n"
                answer += f"🔗 {url}\n"

                # Показываем релевантность если она низкая
                if relevance < 0.7:
                    answer += f"_Релевантность: {relevance:.0%}_\n"

                answer += "\n"

            if len(filtered_results) > 5:
                answer += f"_И ещё {len(filtered_results) - 5} результатов..._"

            return {
                'strategy': 'web_search',
                'confidence': 0.85,
                'answer': answer,
                'sources': filtered_results
            }
        except Exception as e:
            print(f"⚠️ Ошибка веб-поиска: {e}")
            traceback.print_exc()
            return {
                'strategy': 'web_search',
                'confidence': 0.0,
                'answer': f'❌ Ошибка при поиске: {str(e)[:100]}'
            }

    def _create_english_currency_query(self, query: str) -> str:
        """Создание английского запроса для курсов валют"""
        query_lower = query.lower()

        if 'доллар' in query_lower:
            return "USD RUB exchange rate Central Bank Russia today"
        elif 'евро' in query_lower:
            return "EUR RUB exchange rate CBR official"
        elif 'юан' in query_lower:
            return "CNY RUB exchange rate Bank of Russia"
        else:
            return "currency exchange rate Central Bank Russia official"

    def _filter_relevant_results(
            self,
            results: List[Dict[str, Any]],
            original_query: str,
            optimized_query: str
    ) -> List[Dict[str, Any]]:
        """
        УСИЛЕННАЯ фильтрация результатов по релевантности
        Агрессивно удаляет криптовалютные форумы и нерелевантные сайты
        """
        query_lower = original_query.lower()
        optimized_lower = optimized_query.lower()

        # Извлекаем ключевые слова из запроса
        query_keywords = set(MemoryUtils.extract_keywords(original_query, top_n=10))
        query_keywords.update(MemoryUtils.extract_keywords(optimized_query, top_n=10))

        # Специальные ключевые слова для важных тем
        important_keywords = {
            'цб', 'цбр', 'цб рф', 'центробанк', 'центральный банк', 'cbr',
            'official', 'официальный', 'finance', 'финанс',
            'доллар', 'рубл', 'валют', 'евро', 'usd', 'rub', 'eur',
            'погода', 'temperature', 'weather', 'новост', 'news',
            'цена', 'price', 'стоимость', 'cost'
        }

        # КРИТИЧЕСКИЙ ЧЕРНЫЙ СПИСОК - АГРЕССИВНОЕ ОТСЕЧЕНИЕ
        ultra_blacklist = {
            'kurs.expert', 'forum.kurs', 'форум.курс',  # Криптовалютный форум
            'bitcoin', 'биткоин', 'btc', 'крипто', 'crypto',  # Криптовалюты
            'binance', 'coinup', 'обмен криптовалют', 'майнинг',
            'discord', 'download', 'steam', 'game',
            'stackexchange', 'stackoverflow', 'reddit', 'github',
            'softonic', 'tamindir', 'apps.microsoft',
            'telegram', 'whatsapp', 'viber'
        }

        filtered = []
        rejected_count = 0

        for result in results:
            title = result['title'].lower()
            snippet = result['snippet'].lower()
            url = result['url'].lower()

            # НЕМЕДЛЕННОЕ ОТСЕЧЕНИЕ по ультра-черному списку
            is_ultra_blacklisted = any(bl in url or bl in title or bl in snippet for bl in ultra_blacklist)

            if is_ultra_blacklisted:
                rejected_count += 1
                print(f"  ✗ ОТКЛОНЕНО (blacklist): {result['title'][:60]}")
                continue  # Пропускаем без дальнейшей обработки

            # Объединяем текст для анализа
            full_text = f"{title} {snippet}"

            # Вычисляем релевантность
            relevance_score = 0.0

            # 1. ЦЕЛЕВЫЕ ДОМЕНЫ (белый список) - МАКСИМАЛЬНЫЙ ПРИОРИТЕТ
            whitelist_domains = {
                'cbr.ru': 1.5,  # ЦБ РФ - максимальный приоритет!
                'finance.mail.ru': 1.2,
                'banki.ru': 1.2,
                'profinance.ru': 1.0,
                'vbr.ru': 1.0,
                'investing.com': 0.9,
                'rate.ru': 0.9,
                'forexam.ru': 0.8,
                'rbc.ru': 0.8,
                'interfax.ru': 0.8
            }

            domain_bonus = 0.0
            for domain, bonus in whitelist_domains.items():
                if domain in url:
                    domain_bonus = bonus
                    break

            relevance_score += domain_bonus

            # 2. Прямое совпадение важных ключевых слов
            matched_important = sum(1 for kw in important_keywords if kw in full_text)
            if matched_important > 0:
                relevance_score += matched_important * 0.25

            # 3. Совпадение ключевых слов запроса
            text_keywords = set(MemoryUtils.extract_keywords(full_text, top_n=15))
            common_keywords = query_keywords & text_keywords
            if common_keywords:
                relevance_score += len(common_keywords) * 0.12

            # 4. СПЕЦИАЛЬНАЯ ПРОВЕРКА ДЛЯ КУРСОВ ВАЛЮТ
            if 'курс' in query_lower and 'валют' in query_lower or 'доллар' in query_lower or 'евро' in query_lower:
                # ОБЯЗАТЕЛЬНЫЕ слова для курсов валют
                required_currency_words = ['цб', 'официальн', 'банк', 'cbr', 'рубл', 'курс', 'usd', 'rub']
                currency_matches = sum(1 for word in required_currency_words if word in full_text)

                # НЕДОПУСТИМЫЕ слова для курсов валют
                forbidden_words = ['крипто', 'bitcoin', 'btc', 'форум', 'forum', 'обмен']
                forbidden_matches = sum(1 for word in forbidden_words if word in full_text)

                if currency_matches >= 2 and forbidden_matches == 0:
                    relevance_score += 0.5  # Большой бонус за релевантность
                elif forbidden_matches > 0:
                    relevance_score -= 1.0  # Большой штраф за криптовалюты
                elif currency_matches == 0:
                    relevance_score -= 0.5  # Штраф за отсутствие ключевых слов

            # 5. Проверка наличия ЦБ РФ в URL или тексте (для курсов валют)
            if any(indicator in url or indicator in title for indicator in
                   ['cbr.ru', 'цб', 'центробанк', 'central-bank']):
                relevance_score += 0.8

            # Нормализуем score
            relevance_score = max(0.0, min(2.0, relevance_score))

            # ПОВЫШЕННЫЙ ПОРОГ ОТСЕЧЕНИЯ для более строгой фильтрации
            if relevance_score >= 0.4:
                result['relevance_score'] = relevance_score
                filtered.append(result)
                print(f"  ✓ ПРИНЯТО: {result['title'][:60]} | score: {relevance_score:.2f}")
            else:
                rejected_count += 1
                print(f"  ✗ ОТКЛОНЕНО (low score {relevance_score:.2f}): {result['title'][:60]}")

        # Сортируем по релевантности
        filtered.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        print(f"📊 Результат фильтрации: {len(filtered)} принято / {rejected_count} отклонено из {len(results)} всего")

        return filtered

    def _create_simple_query(self, query: str) -> str:
        """
        Создаёт АЛЬТЕРНАТИВНЫЙ упрощенный поисковый запрос
        Используется если основной запрос не дал результатов
        """
        query_lower = query.lower()

        # СПЕЦИАЛИЗИРОВАННЫЕ АЛЬТЕРНАТИВЫ для курсов валют
        if 'курс' in query_lower:
            if 'доллар' in query_lower:
                # Множественные варианты поиска
                alternatives = [
                    "cbr.ru доллар рубль",
                    "официальный курс USD ЦБ",
                    "банк россии доллар сегодня",
                    "usd rub exchange rate central bank"
                ]
                return alternatives[0]  # Возвращаем первый (лучший)
            elif 'евро' in query_lower:
                return "cbr.ru евро рубль официальный"
            else:
                return "cbr.ru валютные курсы официальные"

        # Для других типов - базовые ключевые слова
        keywords = MemoryUtils.extract_keywords(query, top_n=3)
        return ' '.join(keywords)

    async def _handle_web_search_deep(self, query: str) -> Dict[str, Any]:
        """
        Глубокий веб-поиск с загрузкой содержимого и анализом через LLM
        """
        try:
            search_query = self._optimize_search_query(query)
            print(f"🔎 Глубокий поиск: {search_query}")

            # Выполняем поиск с загрузкой содержимого
            search_result = await self.web_searcher.search(
                search_query,
                max_results=5,
                deep_fetch=True
            )

            if not search_result['success'] or not search_result['results']:
                return {
                    'strategy': 'web_search_deep',
                    'confidence': 0.0,
                    'answer': f'❌ Не удалось найти информацию по запросу: "{query}"'
                }

            results = search_result['results']

            # Собираем контент со всех страниц
            sources_with_content = [r for r in results if r.get('has_content')]

            if not sources_with_content:
                # Если не удалось загрузить содержимое, возвращаемся к обычному поиску
                return await self._handle_web_search(query)

            # Создаём синтез из найденной информации
            synthesis = await self.memory.creative_synthesis.synthesize_from_sources(
                sources_with_content,
                synthesis_type='analytical'
            )

            # Формируем промпт для LLM с собранной информацией
            llm_prompt = f"""На основе следующих источников информации ответь на вопрос: "{query}"

ИСТОЧНИКИ:
"""
            for i, source in enumerate(sources_with_content[:3], 1):
                content = source.get('content', source.get('snippet', ''))[:1500]
                llm_prompt += f"\n{i}. {source['title']}\n{content}\n"

            if synthesis:
                llm_prompt += f"\n\nАНАЛИТИЧЕСКИЙ СИНТЕЗ:\n{synthesis['synthesis_text']}"

            llm_prompt += "\n\nОТВЕТ: Проанализируй информацию и дай структурированный ответ."

            # Генерируем умный ответ через LLM
            llm_response = await self.llm.generate(
                llm_prompt,
                temperature=0.5,
                max_tokens=1500
            )

            # Формируем итоговый ответ
            answer = f"🧠 **Анализ из {len(sources_with_content)} источников:**\n\n"
            answer += llm_response
            answer += "\n\n📚 **Источники:**\n"

            for i, source in enumerate(sources_with_content[:3], 1):
                answer += f"{i}. {source['title'][:80]}\n   🔗 {source['url']}\n"

            if synthesis:
                answer += f"\n{synthesis['synthesis_text']}"

            # Сохраняем синтез в памяти
            if synthesis:
                if 'syntheses_created' not in self.memory.metadata:
                    self.memory.metadata['syntheses_created'] = 0
                self.memory.metadata['syntheses_created'] += 1

            return {
                'strategy': 'web_search_deep',
                'confidence': 0.95,
                'answer': answer,
                'sources': sources_with_content,
                'synthesis': synthesis
            }

        except Exception as e:
            print(f"⚠️ Ошибка глубокого поиска: {e}")
            traceback.print_exc()
            # Fallback к обычному поиску
            return await self._handle_web_search(query)

    def _optimize_search_query(self, query: str) -> str:
        """
        УЛУЧШЕННАЯ оптимизация поискового запроса для DuckDuckGo
        Ключевая проблема: DuckDuckGo плохо понимает контекст, нужны ТОЧНЫЕ ключевые слова
        """
        query_lower = query.lower()

        # КРИТИЧЕСКИ ВАЖНО: Определяем ТИП запроса ПЕРЕД очисткой
        query_type = self._detect_query_type(query_lower)

        # Убираем вводные слова и команды
        remove_words = [
            'найди', 'поищи', 'покажи', 'скажи', 'расскажи',
            'пожалуйста', 'можешь', 'ты', 'мне',
            'про', 'о', 'об', 'проведи', 'глубокий', 'анализ',
            'актуальный', 'свежий', 'последний'
        ]

        words = query.split()
        optimized_words = [w for w in words if w.lower() not in remove_words]
        optimized_query = ' '.join(optimized_words).strip()

        # Если после очистки запрос пустой, используем тип запроса
        if not optimized_query or len(optimized_query) < 3:
            if query_type:
                return query_type
            return query

        # СПЕЦИАЛИЗИРОВАННАЯ ОБРАБОТКА ПО ТИПУ ЗАПРОСА

        if query_type == 'currency_rate':
            return self._optimize_currency_query(query_lower)
        elif query_type == 'weather':
            return self._optimize_weather_query(query_lower, optimized_query)
        elif query_type == 'news':
            return self._optimize_news_query(query_lower, optimized_query)
        elif query_type == 'price':
            return self._optimize_price_query(query_lower, optimized_query)

        return optimized_query.strip()

    def _detect_query_type(self, query_lower: str) -> Optional[str]:
        """Определение типа запроса"""
        if 'курс' in query_lower and any(curr in query_lower for curr in ['доллар', 'евро', 'юан', 'фунт', 'валют']):
            return 'currency_rate'
        elif 'погода' in query_lower or 'weather' in query_lower:
            return 'weather'
        elif 'новост' in query_lower or 'news' in query_lower:
            return 'news'
        elif 'цена' in query_lower or 'стоимость' in query_lower or 'price' in query_lower:
            return 'price'
        return None

    def _optimize_currency_query(self, query_lower: str) -> str:
        """
        КРИТИЧЕСКАЯ ОПТИМИЗАЦИЯ для курсов валют
        Проблема: слово "курс" вызывает результаты о форуме kurs.expert
        Решение: использовать альтернативные формулировки
        """
        # Определяем валюту
        currency_map = {
            'доллар': ('доллара', 'USD', 'usd'),
            'евро': ('евро', 'EUR', 'eur'),
            'юан': ('юаня', 'CNY', 'cny'),
            'фунт': ('фунта', 'GBP', 'gbp'),
            'йен': ('йены', 'JPY', 'jpy')
        }

        found_currency = None
        currency_name = None

        for curr_key, (curr_name, curr_code, _) in currency_map.items():
            if curr_key in query_lower:
                found_currency = curr_code
                currency_name = curr_name
                break

        # Если валюта не найдена, ищем общий запрос
        if not found_currency:
            # Используем АЛЬТЕРНАТИВНЫЕ формулировки без слова "курс"
            return "официальный валютный курс ЦБ РФ сегодня"

        # СТРАТЕГИЯ: Использовать разные формулировки для избежания форума kurs.expert
        # Вариант 1: Через официальный источник
        query_variants = [
            f"официальный курс {currency_name} ЦБ РФ сегодня",
            f"{found_currency} рубль ЦБ России",
            f"cbr.ru курс {found_currency}",
            f"Центробанк курс {currency_name}",
            f"exchange rate {found_currency} RUB CBR"
        ]

        # Возвращаем первый вариант (самый точный)
        return query_variants[0]

    def _optimize_weather_query(self, query_lower: str, base_query: str) -> str:
        """Оптимизация запроса о погоде"""
        if 'сегодня' not in query_lower and 'завтра' not in query_lower:
            return base_query + ' сегодня'
        return base_query

    def _optimize_news_query(self, query_lower: str, base_query: str) -> str:
        """Оптимизация запроса о новостях"""
        now = datetime.now()
        if 'сегодня' not in query_lower and 'свежи' not in query_lower:
            return f"{base_query} {now.strftime('%d.%m.%Y')}"
        return base_query

    def _optimize_price_query(self, query_lower: str, base_query: str) -> str:
        """Оптимизация запроса о ценах"""
        if 'купить' not in query_lower:
            return base_query + ' купить'
        return base_query

    async def _handle_memory_search(self, query: str) -> Dict[str, Any]:
        """Поиск в памяти"""
        results = await self.memory.search_memory(query, search_in='all', limit=5)

        if not results:
            return {
                'strategy': 'memory_search',
                'confidence': 0.0,
                'answer': '🤔 Ничего не найдено в моей памяти по этому запросу'
            }

        answer = f"🧠 **Найдено в памяти:** `{query}`\n\n"

        for i, mem in enumerate(results, 1):
            source = mem.get('source', 'unknown')
            score = mem.get('score', 0.0)
            content = mem.get('content', '')[:200]

            source_emoji = {
                'long_term': '📚',
                'episodic': '📖',
                'associative': '🌐'
            }.get(source, '❓')

            answer += f"{i}. {source_emoji} [{source}] 📊 {score:.2f}\n"
            answer += f"   {content}\n\n"

        return {
            'strategy': 'memory_search',
            'confidence': 0.8,
            'answer': answer,
            'results': results
        }

    async def _handle_imagination(self, query: str) -> Dict[str, Any]:
        """Генерация воображаемых идей"""
        imagination = await self.memory.imagine_from_query(query)

        if not imagination:
            return await self._handle_llm_generation(query)

        answer = f"💭 **Воображение:**\n\n"
        answer += f"🧩 **Основа:** {', '.join(imagination['base_concepts'])}\n"
        answer += f"💡 **Новая идея:** {imagination['imagined_concept']}\n"
        answer += f"📊 **Сила связи:** {'⭐' * min(5, int(imagination['strength'] * 5))}\n\n"
        answer += f"📝 {imagination['description']}"

        return {
            'strategy': 'imagination',
            'confidence': imagination['strength'],
            'answer': answer,
            'imagination': imagination
        }

    async def _handle_synthesis(self, query: str) -> Dict[str, Any]:
        """
        Синтез информации из разных источников (память + веб)
        """
        try:
            # 1. Ищем в памяти
            memory_results = await self.memory.search_memory(query, search_in='all', limit=3)

            # 2. Ищем в вебе с глубокой загрузкой
            search_query = self._optimize_search_query(query)
            web_result = await self.web_searcher.search(
                search_query,
                max_results=3,
                deep_fetch=True
            )

            # 3. Объединяем источники
            all_sources = []

            # Добавляем результаты из памяти
            for mem in memory_results:
                all_sources.append({
                    'source': 'memory',
                    'content': mem['content'],
                    'title': f"Память: {mem.get('source', 'unknown')}",
                    'score': mem.get('score', 0.5)
                })

            # Добавляем результаты из веба
            if web_result['success']:
                for web_item in web_result['results']:
                    if web_item.get('has_content'):
                        all_sources.append(web_item)

            if len(all_sources) < 2:
                return {
                    'strategy': 'synthesis',
                    'confidence': 0.3,
                    'answer': '⚠️ Недостаточно источников для синтеза. Нужно минимум 2 источника.'
                }

            # 4. Создаём синтез
            synthesis = await self.memory.creative_synthesis.synthesize_from_sources(
                all_sources,
                synthesis_type='creative'
            )

            if not synthesis:
                return {
                    'strategy': 'synthesis',
                    'confidence': 0.3,
                    'answer': '❌ Не удалось создать синтез'
                }

            # 5. Генерируем ответ через LLM
            llm_prompt = f"""Создай творческий синтез информации на тему: "{query}"

ДОСТУПНЫЕ ИСТОЧНИКИ ({len(all_sources)}):
"""
            for i, source in enumerate(all_sources, 1):
                content = source.get('content', source.get('snippet', ''))[:800]
                source_type = source.get('source', 'unknown')
                llm_prompt += f"\n{i}. [{source_type}] {source.get('title', 'Без названия')}\n{content}\n"

            llm_prompt += f"\n\nКЛЮЧЕВЫЕ ТЕМЫ: {', '.join(synthesis['common_themes'])}"
            llm_prompt += f"\nУНИКАЛЬНЫЕ АСПЕКТЫ: {len(synthesis['unique_aspects'])}"

            llm_prompt += "\n\nЗАДАНИЕ: Создай целостный, творческий ответ, объединяющий информацию из всех источников. Найди неожиданные связи и инсайты."

            llm_response = await self.llm.generate(
                llm_prompt,
                temperature=0.7,
                max_tokens=1500
            )

            # 6. Формируем итоговый ответ
            answer = f"🎨 **ТВОРЧЕСКИЙ СИНТЕЗ**\n\n"
            answer += llm_response
            answer += f"\n\n📊 **Анализ:**\n"
            answer += synthesis['synthesis_text']
            answer += f"\n\n📚 Синтезировано из {len(all_sources)} источников "
            answer += f"({sum(1 for s in all_sources if s.get('source') == 'memory')} из памяти, "
            answer += f"{sum(1 for s in all_sources if s.get('source') != 'memory')} из веба)"

            # Сохраняем
            if 'syntheses_created' not in self.memory.metadata:
                self.memory.metadata['syntheses_created'] = 0
            self.memory.metadata['syntheses_created'] += 1

            return {
                'strategy': 'synthesis',
                'confidence': 0.9,
                'answer': answer,
                'synthesis': synthesis,
                'sources': all_sources
            }

        except Exception as e:
            print(f"⚠️ Ошибка синтеза: {e}")
            traceback.print_exc()
            return await self._handle_llm_generation(query)

    async def _handle_llm_generation(self, query: str) -> Dict[str, Any]:
        """Генерация ответа через LLM"""
        context = self.memory.get_context_for_llm(max_tokens=1500)

        system_prompt = f"""Ты - умный Telegram-бот с системой памяти, созданный для помощи пользователям.

ТВОИ ВОЗМОЖНОСТИ:
• Помнишь предыдущие разговоры (краткосрочная и долгосрочная память)
• Ищешь информацию в интернете (через команды с ключевыми словами)
• Анализируешь и запоминаешь важные факты
• Создаёшь ассоциации между концептами
• Синтезируешь информацию из разных источников
• Можешь сообщить текущее время и дату

ЧТО ТЫ НЕ МОЖЕШЬ:
• НЕ можешь напрямую получать данные из интернета в реальном времени
• НЕ имеешь доступа к актуальным курсам валют, погоде или новостям без веб-поиска
• НЕ можешь выполнять код или открывать сайты
• НЕ имеешь информации после января 2025 года без веб-поиска

{context}

ВАЖНО: Если пользователь просит актуальную информацию (курс валют, погода, новости), 
объясни, что для этого нужен веб-поиск, и предложи переформулировать запрос с 
ключевыми словами типа "найди", "курс", "погода", "новости".

Для глубокого анализа нескольких источников используй слова "сравни", "анализ", "обзор".
Для творческого синтеза используй слово "синтезируй".

Отвечай честно, не выдумывай информацию. Будь полезным и естественным."""

        response = await self.llm.generate(
            prompt=query,
            system=system_prompt,
            temperature=0.7,
            max_tokens=2000
        )

        return {
            'strategy': 'llm_generation',
            'confidence': 0.7,
            'answer': response
        }


# ==================== TELEGRAM БОТ ====================
class TelegramBot:
    """Telegram-интерфейс бота"""

    def __init__(self):
        self.llm = LLMInterface(LM_STUDIO_API_URL, LM_STUDIO_API_KEY)
        self.web_searcher = EnhancedWebSearcher()
        self.managers: Dict[str, QueryManager] = {}

    async def initialize(self):
        """Инициализация"""
        await self.llm.initialize()
        await self.web_searcher.initialize()
        print("✅ TelegramBot инициализирован")

    async def close(self):
        """Закрытие"""
        await self.llm.close()
        await self.web_searcher.close()

    def get_manager(self, user_id: str) -> QueryManager:
        """Получение или создание менеджера для пользователя"""
        if user_id not in self.managers:
            self.managers[user_id] = QueryManager(user_id, self.llm, self.web_searcher)
        return self.managers[user_id]

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start"""
        time_info = TimeUtils.get_current_time_info()
        time_of_day = TimeUtils.get_time_of_day()

        greeting = {
            'утро': 'Доброе утро',
            'день': 'Добрый день',
            'вечер': 'Добрый вечер',
            'ночь': 'Доброй ночи'
        }.get(time_of_day, 'Привет')

        await update.message.reply_text(
            f"{greeting}! 👋\n\n"
            f"🧠 **Мини-Мозг Бот v10.0**\n\n"
            f"✨ **Что нового:**\n"
            f"• 🔍 Улучшенный веб-поиск с интеллектуальным анализом\n"
            f"• 🎨 Система творческого синтеза информации\n"
            f"• 🧠 Глубокий анализ через LLM\n"
            f"• 💭 Расширенное воображение и объединение идей\n"
            f"• ⚡ Оптимизированный кеш с адаптивным TTL\n\n"
            f"📅 Сегодня: {time_info['datetime'].strftime('%d.%m.%Y')}, {time_info['weekday_name']}\n"
            f"⏰ Время: {time_info['datetime'].strftime('%H:%M')}\n\n"
            f"💡 **Примеры запросов:**\n"
            f"• `найди курс доллара` - обычный поиск\n"
            f"• `сравни iPhone и Samsung` - глубокий анализ\n"
            f"• `синтезируй информацию о Python` - творческий синтез\n\n"
            f"Используй /help для полной справки",
            parse_mode='Markdown'
        )

    async def memory_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Статистика памяти"""
        user_id = str(update.effective_user.id)
        manager = self.get_manager(user_id)

        stats = manager.memory.get_stats()

        msg = (
            f"🧠 **СТАТИСТИКА ПАМЯТИ**\n\n"
            f"📝 Краткосрочная: {stats['short_term_count']} сообщений\n"
            f"📚 Долгосрочная: {stats['long_term_count']} фактов\n"
            f"📖 Эпизодическая: {stats['episodic_count']} эпизодов\n"
            f"🌐 Ассоциации: {stats['associations']} связей\n"
            f"💭 Воображения: {stats['imaginations']}\n"
            f"🎨 Синтезы: {stats['syntheses']}\n\n"
            f"📊 **Статистика:**\n"
            f"• Всего сообщений: {stats['total_messages']}\n"
            f"• Извлечено фактов: {stats['facts_extracted']}\n"
            f"• Выявлено паттернов: {stats['patterns_identified']}\n"
            f"• Автопереносов в LT: {stats['auto_transfers']}\n"
            f"• Циклов обучения: {stats['learning_cycles']}\n\n"
            f"💬 **Ваш профиль:**\n"
            f"• Стиль: {stats['communication_style']}\n"
            f"• Скорость обучения: {stats['learning_speed']:.2f}\n"
        )

        if stats['frequent_topics']:
            topics = ", ".join(stats['frequent_topics'][:5])
            msg += f"• Частые темы: {topics}\n"

        if stats['cognitive_patterns']:
            patterns = ", ".join([f"{k}({v})" for k, v in list(stats['cognitive_patterns'].items())[:3]])
            msg += f"• Когнитивные паттерны: {patterns}\n"

        await update.message.reply_text(msg, parse_mode='Markdown')

    async def search_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Поиск в памяти"""
        user_id = str(update.effective_user.id)
        manager = self.get_manager(user_id)
        query = ' '.join(context.args) if context.args else ''

        if not query:
            await update.message.reply_text(
                "🔍 Использование: `/search_memory ваш запрос`",
                parse_mode='Markdown'
            )
            return

        results = await manager.memory.search_memory(query, search_in='all', limit=5)

        if not results:
            await update.message.reply_text(
                f"❌ Ничего не найдено по запросу: `{query}`",
                parse_mode='Markdown'
            )
            return

        msg = f"🔍 **Найдено:** `{query}`\n\n"
        for i, mem in enumerate(results, 1):
            source = mem.get('source', 'unknown')
            score = mem.get('score', 0.0)
            msg += f"{i}. [{source}] 📊 {score:.2f}\n"
            msg += f"   {mem.get('content', '')[:150]}\n\n"

        await update.message.reply_text(msg, parse_mode='Markdown')

    async def imagine(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда воображения"""
        user_id = str(update.effective_user.id)
        manager = self.get_manager(user_id)
        query = ' '.join(context.args) if context.args else ''

        if not query:
            await update.message.reply_text(
                "💭 Использование: `/imagine концепт1 концепт2`",
                parse_mode='Markdown'
            )
            return

        imagination = await manager.memory.imagine_from_query(query)

        if not imagination:
            await update.message.reply_text(
                "❌ Не удалось создать воображение. Нужно больше концептов в памяти.",
                parse_mode='Markdown'
            )
            return

        msg = (
            f"💭 **ВООБРАЖЕНИЕ**\n\n"
            f"🧩 Основа: {', '.join(imagination['base_concepts'])}\n"
            f"💡 Идея: {imagination['imagined_concept']}\n"
            f"📊 Сила: {'⭐' * min(5, int(imagination['strength'] * 5))}\n\n"
            f"📝 {imagination['description']}"
        )

        await update.message.reply_text(msg, parse_mode='Markdown')

    async def syntheses(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать недавние синтезы"""
        user_id = str(update.effective_user.id)
        manager = self.get_manager(user_id)

        recent = manager.memory.creative_synthesis.get_recent_syntheses(5)

        if not recent:
            await update.message.reply_text(
                "📭 Пока нет сохранённых синтезов",
                parse_mode='Markdown'
            )
            return

        msg = "🎨 **НЕДАВНИЕ СИНТЕЗЫ:**\n\n"
        for i, syn in enumerate(recent, 1):
            msg += f"{i}. **{syn['type']}** ({syn['sources_count']} источников)\n"
            msg += f"   {syn['synthesis_text'][:150]}...\n\n"

        await update.message.reply_text(msg, parse_mode='Markdown')

    async def clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Очистка краткосрочной памяти"""
        user_id = str(update.effective_user.id)
        manager = self.get_manager(user_id)

        manager.memory.short_term = []
        manager.memory.working_memory['conversation_context'] = []
        manager.memory._save_all()

        await update.message.reply_text(
            "🧹 **Краткосрочная память очищена!**\n"
            "Долгосрочная, эпизодическая и ассоциативная память сохранены.",
            parse_mode='Markdown'
        )

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Справка"""
        await update.message.reply_text(
            "📖 **СПРАВКА - МИНИ-МОЗГ БОТ v10.0**\n\n"
            "🧠 **Система памяти:**\n"
            "• Краткосрочная - последние сообщения\n"
            "• Долгосрочная - важные факты\n"
            "• Эпизодическая - воспоминания\n"
            "• Ассоциативная - связи между идеями\n"
            "• Творческий синтез - объединение информации\n\n"
            "🔍 **Типы поиска:**\n"
            "• Обычный: `найди курс доллара`\n"
            "• Глубокий: `сравни Python и JavaScript`\n"
            "• Синтез: `синтезируй информацию о AI`\n\n"
            "💡 **Ключевые слова:**\n"
            "• найди, поищи - обычный поиск\n"
            "• сравни, анализ, обзор - глубокий анализ\n"
            "• синтезируй, объедини - творческий синтез\n"
            "• представь, придумай - воображение\n\n"
            "📝 **Команды:**\n"
            "/memory - статистика памяти\n"
            "/search_memory <запрос> - поиск в памяти\n"
            "/imagine <концепты> - воображение\n"
            "/syntheses - недавние синтезы\n"
            "/clear - очистка краткосрочной памяти\n"
            "/help - эта справка",
            parse_mode='Markdown'
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка сообщений"""
        user_id = str(update.effective_user.id)
        text = update.message.text.strip()

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        try:
            manager = self.get_manager(user_id)

            if not hasattr(manager, '_initialized'):
                await manager.initialize()
                manager._initialized = True

            result = await manager.process_query(text)

            answer = result.get('answer', 'Не удалось сформировать ответ')

            if len(answer) > 4000:
                answer = answer[:3950] + "\n\n...(усечено)"

            await update.message.reply_text(
                answer,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )

            strategy = result.get('strategy', 'unknown')
            confidence = result.get('confidence', 0)
            print(f"✅ user_{user_id} | {strategy} | {confidence:.2f}")

        except Exception as e:
            error_msg = f"❌ **Ошибка:** {str(e)[:200]}"
            print(f"ERROR user_{user_id}: {e}")
            traceback.print_exc()

            await update.message.reply_text(error_msg, parse_mode='Markdown')


# ==================== ЗАПУСК ====================
async def main_async():
    """Асинхронная главная функция"""
    print("\n" + "=" * 70)
    print("🚀 МИНИ-МОЗГ БОТ v10.0 - УЛУЧШЕННАЯ ВЕРСИЯ")
    print("=" * 70)

    time_info = TimeUtils.get_current_time_info()
    print(f"⏰ Запуск: {time_info['formatted']}")
    print(f"📅 Дата: {time_info['datetime'].strftime('%d.%m.%Y')}, {time_info['weekday_name']}")
    print(f"🧠 Память: {MEMORY_DIR}/")
    print(f"🌐 Веб-поиск: {'✅ Async' if DDGS_ASYNC else '✅ Sync' if DDGS_AVAILABLE else '❌'}")
    print(f"🍲 BeautifulSoup: {'✅' if BEAUTIFULSOUP_AVAILABLE else '❌'}")
    print("=" * 70)

    # Проверка LM Studio
    try:
        async with aiohttp.ClientSession() as session:
            test_url = LM_STUDIO_API_URL.replace('/v1/chat/completions', '/v1/models')
            async with session.get(test_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    print(f"✅ LM Studio доступна")
                else:
                    print(f"⚠️ LM Studio код: {resp.status}")
    except Exception as e:
        print(f"⚠️ LM Studio недоступна: {type(e).__name__}")

    print("=" * 70)
    print("\n🔄 Инициализация...")

    bot = TelegramBot()
    await bot.initialize()

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_cmd))
    application.add_handler(CommandHandler("memory", bot.memory_stats))
    application.add_handler(CommandHandler("search_memory", bot.search_memory))
    application.add_handler(CommandHandler("imagine", bot.imagine))
    application.add_handler(CommandHandler("syntheses", bot.syntheses))
    application.add_handler(CommandHandler("clear", bot.clear))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    print("✅ Бот готов!")
    print("=" * 70)
    print("🎯 УЛУЧШЕНИЯ v10.0:")
    print("   • 🔍 Интеллектуальный веб-поиск с адаптивным кешем")
    print("   • 📥 Глубокая загрузка содержимого страниц")
    print("   • 🧠 Анализ результатов через LLM")
    print("   • 🎨 Система творческого синтеза информации")
    print("   • 💭 Расширенное воображение и кросс-опыление идей")
    print("   • ⚡ Оптимизация и TTL кеша")
    print("   • 🌐 Интеграция памяти и веб-поиска")
    print("=" * 70)
    print("\nCtrl+C для остановки\n")

    try:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)

        while True:
            await asyncio.sleep(3600)

    except KeyboardInterrupt:
        print("\n🛑 Остановка...")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        traceback.print_exc()
    finally:
        await application.stop()
        await application.shutdown()
        await bot.close()
        print("✅ Бот остановлен")


def main():
    """Точка входа"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n👋 Выход")
    except Exception as e:
        print(f"\n❌ Фатальная ошибка: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()