# test_lm_studio.py
import asyncio
import aiohttp
import json

async def test():
    url = "http://127.0.0.1:1234/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": "Привет! Ответь одним словом."}],
        "temperature": 0.3,
        "max_tokens": 50,
        "model": "qwen/qwen3.5-9b",
        "stream": False
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            text = await resp.text()
            print(f"Status: {resp.status}")
            print(f"Response: {text}")
            if resp.status == 200:
                try:
                    data = json.loads(text)
                    if "choices" in data:
                        print("✅ Ответ:", data["choices"][0]["message"]["content"])
                    elif "error" in data:
                        print("❌ Ошибка LM Studio:", data["error"])
                except:
                    print("⚠️ Не JSON")

asyncio.run(test())