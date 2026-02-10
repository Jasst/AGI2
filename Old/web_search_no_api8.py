# coding: utf-8
"""
🔍 ПОЛНОЦЕННЫЙ ИНТЕРНЕТ-ПОИСК БЕЗ API — ИСПРАВЛЕННАЯ ВЕРСИЯ
✅ Корректные User-Agent для всех запросов
✅ Исправлены ошибки в URL
✅ Улучшенное извлечение контента
✅ Обработка ошибок и таймаутов
"""

import aiohttp
import asyncio
import re
import json
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from urllib.parse import quote_plus, urljoin, unquote
import random

# ==================== КОНФИГУРАЦИЯ ====================
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
]

SEARCH_CACHE = {}
CACHE_DURATION = timedelta(hours=1)


# ==================== БАЗОВЫЙ КЛАСС ====================
class SearchEngine:
    """Базовый класс для поисковых движков"""

    def __init__(self):
        self.name = "Base"
        self.timeout = 20

    def get_headers(self) -> Dict[str, str]:
        """Генерация реалистичных заголовков"""
        return {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache'
        }

    async def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Поиск (переопределяется в наследниках)"""
        raise NotImplementedError


# ==================== DUCKDUCKGO ====================
class DuckDuckGoSearch(SearchEngine):
    """Поиск через DuckDuckGo HTML"""

    def __init__(self):
        super().__init__()
        self.name = "DuckDuckGo"
        self.base_url = "https://html.duckduckgo.com/html/"  # Исправлено: убраны лишние пробелы

    async def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Поиск через HTML версию DDG"""
        results = []

        try:
            async with aiohttp.ClientSession(headers=self.get_headers()) as session:
                # POST запрос как в браузере
                data = {
                    'q': query,
                    'b': '',
                    'kl': 'ru-ru',
                    's': '0'
                }

                async with session.post(
                        self.base_url,
                        data=data,
                        timeout=self.timeout
                ) as resp:
                    if resp.status != 200:
                        print(f"DDG ошибка: статус {resp.status}")
                        return results

                    html = await resp.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Парсинг результатов
                    for result in soup.select('.result')[:limit]:
                        try:
                            title_elem = result.select_one('.result__a')
                            snippet_elem = result.select_one('.result__snippet')

                            if title_elem:
                                title = title_elem.get_text(strip=True)
                                url = title_elem.get('href', '')
                                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                                # Очистка URL (DDG перенаправляет через свой сервис)
                                if url.startswith('//duckduckgo.com/l/?') or url.startswith(
                                        'https://duckduckgo.com/l/?'):
                                    url_match = re.search(r'uddg=([^&]+)', url)
                                    if url_match:
                                        url = unquote(url_match.group(1))

                                # Убираем пустые URL
                                if url and url != '#':
                                    results.append({
                                        'title': title,
                                        'url': url,
                                        'snippet': snippet,
                                        'source': 'DuckDuckGo'
                                    })
                        except Exception as e:
                            print(f"DDG ошибка парсинга: {e}")
                            continue

        except asyncio.TimeoutError:
            print(f"DDG таймаут")
        except Exception as e:
            print(f"DDG ошибка: {e}")

        return results


# ==================== SEARX (метапоиск) ====================
class SearxSearch(SearchEngine):
    """Поиск через публичные инстансы Searx"""

    def __init__(self):
        super().__init__()
        self.name = "Searx"
        # Публичные инстансы Searx (убраны лишние пробелы)
        self.instances = [
            "https://searx.be",
            "https://search.bus-hit.me",
            "https://searx.tiekoetter.com",
            "https://search.sapti.me",
            "https://searx.privacyguides.net"
        ]

    async def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Поиск через Searx API"""
        results = []

        for instance in self.instances:
            try:
                async with aiohttp.ClientSession(headers=self.get_headers()) as session:
                    params = {
                        'q': query,
                        'format': 'json',
                        'language': 'ru',
                        'safesearch': '0',
                        'categories': 'general'
                    }

                    async with session.get(
                            f"{instance}/search",
                            params=params,
                            timeout=15
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()

                            for result in data.get('results', [])[:limit]:
                                results.append({
                                    'title': result.get('title', ''),
                                    'url': result.get('url', ''),
                                    'snippet': result.get('content', ''),
                                    'source': 'Searx'
                                })

                            if results:
                                print(f"Searx: успешно с {instance}")
                                break  # Успешно получили результаты

            except asyncio.TimeoutError:
                print(f"Searx таймаут: {instance}")
                continue
            except Exception as e:
                print(f"Searx ошибка {instance}: {e}")
                continue

        return results


# ==================== ВИКИПЕДИЯ ====================
class WikipediaSearch(SearchEngine):
    """Поиск в Википедии"""

    def __init__(self, lang='ru'):
        super().__init__()
        self.name = f"Wikipedia ({lang})"
        self.lang = lang
        self.api_url = f"https://{lang}.wikipedia.org/w/api.php"

    async def search(self, query: str, limit: int = 3) -> List[Dict]:
        """Поиск через Wikipedia API"""
        results = []

        try:
            async with aiohttp.ClientSession(headers=self.get_headers()) as session:
                # Поиск страниц
                search_params = {
                    'action': 'opensearch',
                    'search': query,
                    'limit': limit,
                    'format': 'json',
                    'redirects': 'resolve',
                    'namespace': '0'
                }

                async with session.get(
                        self.api_url,
                        params=search_params,
                        timeout=self.timeout
                ) as resp:
                    if resp.status != 200:
                        print(f"Wikipedia ошибка: статус {resp.status}")
                        return results

                    data = await resp.json()

                    if len(data) >= 4:
                        titles = data[1]
                        descriptions = data[2]
                        urls = data[3]

                        for title, desc, url in zip(titles, descriptions, urls):
                            if title and url:  # Проверяем что данные есть
                                results.append({
                                    'title': title,
                                    'url': url,
                                    'snippet': desc,
                                    'source': 'Wikipedia'
                                })

        except asyncio.TimeoutError:
            print(f"Wikipedia таймаут")
        except Exception as e:
            print(f"Wikipedia ошибка: {e}")

        return results


# ==================== ЯНДЕКС ====================
class YandexSearch(SearchEngine):
    """Поиск через Яндекс (простой парсинг)"""

    def __init__(self):
        super().__init__()
        self.name = "Yandex"
        self.base_url = "https://yandex.ru/search/"  # Исправлено: убраны лишние пробелы

    async def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Поиск через Яндекс"""
        results = []

        try:
            params = {
                'text': query,
                'lr': '213',  # lr=213 - Москва
                'numdoc': str(limit)
            }

            async with aiohttp.ClientSession(headers=self.get_headers()) as session:
                async with session.get(
                        self.base_url,
                        params=params,
                        timeout=self.timeout
                ) as resp:
                    if resp.status != 200:
                        print(f"Yandex ошибка: статус {resp.status}")
                        return results

                    html = await resp.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Яндекс использует разные селекторы
                    for item in soup.select('.serp-item')[:limit]:
                        try:
                            title_elem = item.select_one('.OrganicTitle-Link, h2 a, .link')
                            snippet_elem = item.select_one('.OrganicTextContentSpan, .text-container, .path')

                            if title_elem:
                                title = title_elem.get_text(strip=True)
                                url = title_elem.get('href', '')
                                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                                # Очистка URL (Yandex тоже перенаправляет)
                                if url.startswith('//'):
                                    url = 'https:' + url
                                elif url.startswith('/'):
                                    url = 'https://yandex.ru' + url

                                if url and url.startswith('http'):
                                    results.append({
                                        'title': title,
                                        'url': url,
                                        'snippet': snippet,
                                        'source': 'Yandex'
                                    })
                        except Exception as e:
                            print(f"Yandex ошибка парсинга: {e}")
                            continue

        except asyncio.TimeoutError:
            print(f"Yandex таймаут")
        except Exception as e:
            print(f"Yandex ошибка: {e}")

        return results


# ==================== ИЗВЛЕЧЕНИЕ КОНТЕНТА ====================
class ContentExtractor:
    """Извлечение основного контента со страницы"""

    @staticmethod
    async def extract_from_url(url: str, max_length: int = 2000) -> Optional[str]:
        """Извлечение текста со страницы"""
        try:
            headers = {
                'User-Agent': random.choice(USER_AGENTS),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            }

            async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status != 200:
                        print(f"Ошибка загрузки {url}: статус {resp.status}")
                        return None

                    html = await resp.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Удаляем скрипты, стили, навигацию
                    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
                        tag.decompose()

                    # Ищем основной контент
                    main_content = (
                            soup.find('article') or
                            soup.find('main') or
                            soup.find('div', class_=re.compile('content|article|post|entry|main')) or
                            soup.find('div', id=re.compile('content|article|main')) or
                            soup.find('body')
                    )

                    if main_content:
                        # Извлекаем параграфы
                        paragraphs = main_content.find_all('p')
                        text = '\n'.join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)

                        # Если нет параграфов, ищем div с текстом
                        if not text or len(text) < 200:
                            divs = main_content.find_all('div')
                            text = '\n'.join(d.get_text(strip=True) for d in divs if len(d.get_text(strip=True)) > 50)

                        # Ограничиваем длину
                        if len(text) > max_length:
                            text = text[:max_length] + "..."

                        # Очищаем лишние пробелы и переносы
                        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
                        text = text.strip()

                        return text if len(text) > 50 else None

        except asyncio.TimeoutError:
            print(f"Таймаут извлечения контента: {url}")
            return None
        except Exception as e:
            print(f"Ошибка извлечения контента {url}: {e}")
            return None


# ==================== МУЛЬТИПОИСК ====================
class MultiSearch:
    """Объединенный поиск через несколько движков"""

    def __init__(self):
        self.engines = [
            DuckDuckGoSearch(),
            SearxSearch(),
            WikipediaSearch(),
            YandexSearch(),
        ]
        self.content_extractor = ContentExtractor()

    async def search(self, query: str, deep: bool = False) -> Dict:
        """
        Поиск через несколько движков

        Args:
            query: поисковый запрос
            deep: если True, извлекает контент со страниц

        Returns:
            Dict с результатами
        """
        # Проверка кэша
        cache_key = f"{query}_{deep}"
        if cache_key in SEARCH_CACHE:
            cached_data, cached_time = SEARCH_CACHE[cache_key]
            if datetime.now() - cached_time < CACHE_DURATION:
                print(f"📦 Кэш найден для: {query}")
                return cached_data

        print(f"🔍 Поиск: {query}")

        # Параллельный поиск через все движки
        tasks = [engine.search(query, limit=3) for engine in self.engines]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Объединение и дедупликация результатов
        all_results = []
        seen_urls = set()

        for results in results_list:
            if isinstance(results, list):
                for result in results:
                    url = result.get('url', '').strip()
                    if url and url not in seen_urls and url.startswith('http'):
                        seen_urls.add(url)
                        all_results.append(result)

        # Глубокий поиск - извлекаем контент со страниц
        if deep and all_results:
            print(f"📄 Извлечение контента с {min(3, len(all_results))} страниц...")

            extract_tasks = [self.content_extractor.extract_from_url(result['url']) for result in all_results[:3]]
            contents = await asyncio.gather(*extract_tasks, return_exceptions=True)

            for result, content in zip(all_results[:3], contents):
                if isinstance(content, str) and content:
                    result['full_content'] = content

        response = {
            'query': query,
            'results': all_results[:10],  # Топ-10
            'total_found': len(all_results),
            'timestamp': datetime.now().isoformat()
        }

        # Кэшируем результат
        SEARCH_CACHE[cache_key] = (response, datetime.now())

        return response

    def format_results(self, search_data: Dict) -> str:
        """Форматирование результатов в читаемый текст"""
        output = f"🔍 Результаты поиска: {search_data['query']}\n"
        output += f"📊 Найдено: {search_data['total_found']} результатов\n\n"

        for i, result in enumerate(search_data['results'][:5], 1):
            output += f"{i}. {result['title']}\n"
            snippet = result.get('snippet', '')
            output += f"   {snippet[:200]}...\n" if len(snippet) > 200 else f"   {snippet}\n"
            output += f"   🔗 {result['url']}\n"
            output += f"   📌 Источник: {result['source']}\n\n"

            # Если есть полный контент
            if 'full_content' in result:
                content = result['full_content']
                output += f"   📄 Контент:\n   {content[:500]}...\n\n"

        return output


# ==================== СПЕЦИАЛИЗИРОВАННЫЕ ПОИСКИ ====================
class SpecializedSearch:
    """Специализированные поиски"""

    @staticmethod
    async def get_bitcoin_price() -> Dict:
        """Получение курса Bitcoin БЕЗ API"""
        try:
            # Парсим с CoinGecko (публичная страница)
            url = "https://www.coingecko.com/en/coins/bitcoin"  # Исправлено: убраны лишние пробелы
            headers = {
                'User-Agent': random.choice(USER_AGENTS),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }

            async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status != 200:
                        print(f"CoinGecko ошибка: статус {resp.status}")
                        return {'error': f'HTTP {resp.status}'}

                    html = await resp.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Ищем цену в HTML (несколько вариантов селекторов)
                    price_elem = None

                    # Вариант 1: через атрибут
                    price_elem = soup.find('span', {'data-coin-symbol': 'btc'})

                    # Вариант 2: через класс
                    if not price_elem:
                        price_elem = soup.select_one('.no-wrap')

                    # Вариант 3: через конкретный класс цены
                    if not price_elem:
                        price_elem = soup.select_one('span[data-target="price.price"]')

                    # Вариант 4: ищем все элементы с $
                    if not price_elem:
                        price_elems = soup.find_all(string=re.compile(r'\$\s*[\d,]+\.?\d*'))
                        if price_elems:
                            price_elem = price_elems[0]

                    if price_elem:
                        price_text = price_elem.get_text(strip=True)
                        # Извлекаем число
                        price_match = re.search(r'[\d,]+\.?\d*', price_text.replace('$', ''))
                        if price_match:
                            price = price_match.group(0).replace(',', '')
                            try:
                                return {
                                    'price_usd': float(price),
                                    'source': 'CoinGecko',
                                    'timestamp': datetime.now().isoformat()
                                }
                            except ValueError:
                                print(f"Ошибка парсинга цены: {price}")
                                return {'error': 'Ошибка парсинга'}

        except asyncio.TimeoutError:
            print(f"Таймаут получения курса BTC")
            return {'error': 'Timeout'}
        except Exception as e:
            print(f"Ошибка получения курса BTC: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

        return {'error': 'Не удалось найти цену'}

    @staticmethod
    async def get_weather(city: str = "Moscow") -> Dict:
        """Получение погоды БЕЗ API"""
        try:
            # Используем wttr.in (бесплатный сервис)
            url = f"https://wttr.in/{city}?format=j1"  # Исправлено: убраны лишние пробелы

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status != 200:
                        print(f"wttr.in ошибка: статус {resp.status}")
                        return {'error': f'HTTP {resp.status}'}

                    data = await resp.json()

                    if 'current_condition' not in data or not data['current_condition']:
                        return {'error': 'Неверные данные'}

                    current = data['current_condition'][0]
                    return {
                        'city': city,
                        'temperature_c': current.get('temp_C', 'N/A'),
                        'feels_like_c': current.get('FeelsLikeC', 'N/A'),
                        'description': current.get('weatherDesc', [{}])[0].get('value', 'N/A'),
                        'humidity': current.get('humidity', 'N/A'),
                        'wind_kph': current.get('windspeedKmph', 'N/A'),
                        'source': 'wttr.in'
                    }
        except asyncio.TimeoutError:
            print(f"Таймаут получения погоды")
            return {'error': 'Timeout'}
        except Exception as e:
            print(f"Ошибка получения погоды: {e}")
            return {'error': str(e)}

    @staticmethod
    async def get_news(topic: str = "россия", limit: int = 5) -> List[Dict]:
        """Поиск новостей через поисковики"""
        # Используем поиск с временным фильтром
        query = f"{topic} новости сегодня"

        multi_search = MultiSearch()
        results = await multi_search.search(query)

        return results['results'][:limit]


# ==================== ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ====================
async def test_search():
    """Тестирование поиска"""
    multi_search = MultiSearch()

    # Простой поиск
    print("=" * 50)
    print("ТЕСТ 1: Простой поиск")
    print("=" * 50)
    results = await multi_search.search("искусственный интеллект")
    print(multi_search.format_results(results))

    # Глубокий поиск с извлечением контента
    print("\n" + "=" * 50)
    print("ТЕСТ 2: Глубокий поиск")
    print("=" * 50)
    results = await multi_search.search("машинное обучение", deep=True)
    print(multi_search.format_results(results))

    # Специализированные поиски
    print("\n" + "=" * 50)
    print("ТЕСТ 3: Курс Bitcoin")
    print("=" * 50)
    btc = await SpecializedSearch.get_bitcoin_price()
    if 'error' in btc:
        print(f"❌ Ошибка: {btc['error']}")
    else:
        print(f"₿ Bitcoin: ${btc.get('price_usd', 'N/A')}")

    print("\n" + "=" * 50)
    print("ТЕСТ 4: Погода")
    print("=" * 50)
    weather = await SpecializedSearch.get_weather("Moscow")
    if 'error' in weather:
        print(f"❌ Ошибка: {weather['error']}")
    else:
        print(f"🌡️ Москва: {weather.get('temperature_c', 'N/A')}°C, {weather.get('description', 'N/A')}")

    print("\n" + "=" * 50)
    print("ТЕСТ 5: Новости")
    print("=" * 50)
    news = await SpecializedSearch.get_news("технологии")
    for i, item in enumerate(news[:3], 1):
        print(f"{i}. {item['title']}")
        print(f"   {item['snippet'][:100]}...")


if __name__ == "__main__":
    asyncio.run(test_search())