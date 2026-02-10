import aiohttp
from bs4 import BeautifulSoup

CACHE = {}


async def search(query: str) -> str:
    if query in CACHE:
        return CACHE[query]

    url = f"https://duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, timeout=20) as resp:
                html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")
        snippets = soup.select(".result__snippet")
        result = "\n".join(s.get_text(strip=True) for s in snippets[:5])

        CACHE[query] = result or "Актуальная информация не найдена."
        return CACHE[query]
    except Exception:
        return "Ошибка интернет-поиска."
