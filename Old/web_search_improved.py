# coding: utf-8
"""
🔍 УЛУЧШЕННЫЙ ИНТЕРНЕТ-ПОИСК v4.1
✨ Новые возможности:
- Больше источников данных
- Интеллектуальное кэширование
- Лучшая обработка ошибок
- Параллельный поиск
- Ранжирование результатов

✅ ИСПРАВЛЕНИЯ:
- Убраны пробелы в конце всех URL
- Исправлен парсинг JSON от ЦБ РФ
- Улучшена обработка ошибок
"""
import aiohttp
import asyncio
import re
import json
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from urllib.parse import unquote, quote
import random
import ssl
import certifi
import hashlib
from collections import defaultdict

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
]

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# Улучшенное кэширование
SEARCH_CACHE = {}
CACHE_DURATION = timedelta(hours=1)

# Расширенный словарь опечаток
TYPO_CORRECTIONS = {
    'даллар': 'доллар', 'далар': 'доллар', 'долар': 'доллар',
    'бтц': 'биткоин', 'бткоин': 'биткоин', 'битеоин': 'биткоин',
    'пагода': 'погода', 'погота': 'погода', 'пагоды': 'погода',
    'интелект': 'интеллект', 'нейроные': 'нейронные',
}


class SearchResult:
    """Класс для хранения результатов поиска с метаданными"""

    def __init__(self, title: str, url: str, snippet: str, source: str):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source
        self.relevance_score = 0.0
        self.timestamp = datetime.now()
        self.full_content = None

    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'url': self.url,
            'snippet': self.snippet,
            'source': self.source,
            'relevance_score': self.relevance_score,
            'timestamp': self.timestamp.isoformat(),
            'full_content': self.full_content
        }


class SearchEngine:
    """Базовый класс для поисковых движков"""

    def __init__(self):
        self.name = "Base"
        self.timeout = 20
        self.max_retries = 2
        self.success_count = 0
        self.fail_count = 0

    def get_headers(self) -> Dict[str, str]:
        return {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
        }

    def get_reliability_score(self) -> float:
        """Вычисляет надежность источника"""
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.5
        return self.success_count / total

    async def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        raise NotImplementedError


class DuckDuckGoSearch(SearchEngine):
    def __init__(self):
        super().__init__()
        self.name = "DuckDuckGo"
        self.base_url = "https://html.duckduckgo.com/html/"  # ✅ ИСПРАВЛЕНО: убран пробел

    async def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        results = []
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(headers=self.get_headers()) as session:
                    data = {'q': query, 'b': '', 'kl': 'ru-ru', 's': '0'}
                    async with session.post(
                            self.base_url,
                            data=data,
                            timeout=self.timeout,
                            ssl=SSL_CONTEXT
                    ) as resp:
                        if resp.status == 200:
                            html = await resp.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            for result in soup.select('.result')[:limit]:
                                try:
                                    title_elem = result.select_one('.result__a')
                                    if title_elem:
                                        title = title_elem.get_text(strip=True)
                                        url = title_elem.get('href', '')
                                        snippet_elem = result.select_one('.result__snippet')
                                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                                        if url.startswith('//duckduckgo.com/l/?'):
                                            url_match = re.search(r'uddg=([^&]+)', url)
                                            if url_match:
                                                url = unquote(url_match.group(1))

                                        if url and url.startswith('http'):
                                            result_obj = SearchResult(title, url, snippet, self.name)
                                            results.append(result_obj)
                                except Exception as e:
                                    continue
                            self.success_count += 1
                            break
                        else:
                            await asyncio.sleep(1 * (attempt + 1))
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.fail_count += 1
                await asyncio.sleep(1 * (attempt + 1))
        return results


class SearxSearch(SearchEngine):
    def __init__(self):
        super().__init__()
        self.name = "Searx"
        self.instances = [
            "https://searx.be",  # ✅ ИСПРАВЛЕНО: убраны пробелы
            "https://search.bus-hit.me",
            "https://searx.name",
            "https://searx.ninja",
            "https://searx.tiekoetter.com",
        ]
        self.working_instances = list(self.instances)

    async def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        results = []
        for instance in self.working_instances[:3]:
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
                            timeout=15,
                            ssl=SSL_CONTEXT
                    ) as resp:
                        if resp.status == 200:
                            content_type = resp.headers.get('Content-Type', '')
                            if 'application/json' in content_type:
                                data = await resp.json()
                                for result in data.get('results', [])[:limit]:
                                    result_obj = SearchResult(
                                        title=result.get('title', ''),
                                        url=result.get('url', ''),
                                        snippet=result.get('content', ''),
                                        source=self.name
                                    )
                                    results.append(result_obj)
                            if results:
                                self.success_count += 1
                                break
            except Exception as e:
                continue

        if not results:
            self.fail_count += 1
        return results


class WikipediaSearch(SearchEngine):
    def __init__(self, lang='ru'):
        super().__init__()
        self.name = f"Wikipedia"
        self.lang = lang
        self.api_url = f"https://{lang}.wikipedia.org/w/api.php"

    async def search(self, query: str, limit: int = 3) -> List[SearchResult]:
        results = []
        try:
            headers = self.get_headers()
            headers['Api-User-Agent'] = 'CognitiveBot/4.0'
            async with aiohttp.ClientSession(headers=headers) as session:
                # Поиск статей
                params = {
                    'action': 'opensearch',
                    'search': query,
                    'limit': limit,
                    'format': 'json'
                }
                async with session.get(
                        self.api_url,
                        params=params,
                        timeout=self.timeout,
                        ssl=SSL_CONTEXT
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if len(data) >= 4:
                            for title, desc, url in zip(data[1], data[2], data[3]):
                                if title and url:
                                    result_obj = SearchResult(
                                        title=title,
                                        url=url,
                                        snippet=desc,
                                        source=self.name
                                    )
                                    results.append(result_obj)
                        self.success_count += 1
        except Exception as e:
            self.fail_count += 1
        return results


class YandexSearch(SearchEngine):
    def __init__(self):
        super().__init__()
        self.name = "Yandex"
        self.base_url = "https://yandex.ru/search/"  # ✅ ИСПРАВЛЕНО: убран пробел

    async def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        results = []
        try:
            params = {'text': query, 'lr': '213', 'numdoc': str(limit)}
            headers = self.get_headers()
            headers['Referer'] = 'https://yandex.ru/'  # ✅ ИСПРАВЛЕНО: убран пробел
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(
                        self.base_url,
                        params=params,
                        timeout=self.timeout,
                        ssl=SSL_CONTEXT
                ) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        for item in soup.select('.serp-item')[:limit]:
                            title_elem = item.select_one('.OrganicTitle-Link')
                            if title_elem:
                                title = title_elem.get_text(strip=True)
                                url = title_elem.get('href', '')
                                snippet_elem = item.select_one('.OrganicTextContentSpan')
                                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                                if url.startswith('//'):
                                    url = 'https:' + url
                                if url.startswith('http'):
                                    result_obj = SearchResult(title, url, snippet, self.name)
                                    results.append(result_obj)
                        if results:
                            self.success_count += 1
        except Exception as e:
            self.fail_count += 1
        return results


class GoogleScholarSearch(SearchEngine):
    """Поиск по Google Scholar для научных статей"""

    def __init__(self):
        super().__init__()
        self.name = "Google Scholar"
        self.base_url = "https://scholar.google.com/scholar"  # ✅ ИСПРАВЛЕНО: убран пробел

    async def search(self, query: str, limit: int = 3) -> List[SearchResult]:
        results = []
        try:
            params = {'q': query, 'hl': 'ru', 'as_sdt': '0,5'}
            headers = self.get_headers()
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(
                        self.base_url,
                        params=params,
                        timeout=self.timeout,
                        ssl=SSL_CONTEXT
                ) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        for item in soup.select('.gs_ri')[:limit]:
                            title_elem = item.select_one('.gs_rt')
                            if title_elem:
                                # Убираем теги
                                for tag in title_elem.find_all(['span', 'b']):
                                    tag.unwrap()
                                title = title_elem.get_text(strip=True)
                                link_elem = title_elem.find('a')
                                url = link_elem.get('href', '') if link_elem else ''
                                snippet_elem = item.select_one('.gs_rs')
                                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                                if url:
                                    result_obj = SearchResult(title, url, snippet, self.name)
                                    results.append(result_obj)
                        if results:
                            self.success_count += 1
        except Exception as e:
            self.fail_count += 1
        return results


class ContentExtractor:
    """Улучшенный экстрактор контента"""

    @staticmethod
    async def extract_from_url(url: str, max_length: int = 3000) -> Optional[str]:
        try:
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            async with aiohttp.ClientSession(
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15)
            ) as session:
                async with session.get(url, timeout=15, ssl=SSL_CONTEXT) as resp:
                    if resp.status == 200:
                        content_type = resp.headers.get('Content-Type', '')
                        # Проверка типа контента
                        if 'text/html' not in content_type:
                            return None
                        html = await resp.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        # Удаление ненужных элементов
                        for tag in soup(['script', 'style', 'nav', 'header', 'footer',
                                         'aside', 'iframe', 'noscript', 'form', 'button']):
                            tag.decompose()
                        # Поиск основного контента
                        main_content = (
                                soup.find('article') or
                                soup.find('main') or
                                soup.find('div', class_=re.compile('content|article|post|entry', re.I)) or
                                soup.find('div', id=re.compile('content|article|post|entry', re.I)) or
                                soup.find('body')
                        )
                        if main_content:
                            # Извлечение параграфов и заголовков
                            elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'li'])
                            text_parts = []
                            for elem in elements:
                                text = elem.get_text(strip=True)
                                if len(text) > 20:  # Фильтр коротких текстов
                                    text_parts.append(text)
                            full_text = '\n'.join(text_parts)
                            if len(full_text) > max_length:
                                full_text = full_text[:max_length] + "..."
                            return full_text.strip() if len(full_text) > 100 else None
        except Exception as e:
            pass
        return None

    @staticmethod
    def extract_key_facts(text: str, query: str) -> List[str]:
        """Извлекает ключевые факты из текста"""
        if not text:
            return []

        # Разбиваем на предложения
        sentences = re.split(r'[.!?]\s+', text)

        # Ищем предложения с ключевыми словами из запроса
        query_words = set(query.lower().split())
        relevant_sentences = []

        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            # Если есть пересечение слов
            if query_words & sentence_words:
                relevant_sentences.append(sentence.strip())

        return relevant_sentences[:5]  # Топ-5 релевантных предложений


class RelevanceRanker:
    """Ранжирование результатов по релевантности"""

    @staticmethod
    def calculate_relevance(result: SearchResult, query: str) -> float:
        """Вычисляет релевантность результата запросу"""
        score = 0.0
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Проверка заголовка
        title_lower = result.title.lower()
        title_words = set(title_lower.split())

        # Точное совпадение в заголовке
        if query_lower in title_lower:
            score += 50.0

        # Совпадение слов в заголовке
        common_title = query_words & title_words
        score += len(common_title) * 10.0

        # Проверка сниппета
        snippet_lower = result.snippet.lower()
        snippet_words = set(snippet_lower.split())

        # Точное совпадение в сниппете
        if query_lower in snippet_lower:
            score += 30.0

        # Совпадение слов в сниппете
        common_snippet = query_words & snippet_words
        score += len(common_snippet) * 5.0

        # Бонус за длину сниппета (более информативные результаты)
        if len(result.snippet) > 100:
            score += 10.0

        # Бонус за надежный источник
        trusted_sources = ['wikipedia', 'gov', 'edu', 'coingecko', 'cbr']
        if any(source in result.url.lower() or source in result.source.lower()
               for source in trusted_sources):
            score += 20.0

        return score

    @staticmethod
    def rank_results(results: List[SearchResult], query: str) -> List[SearchResult]:
        """Ранжирует результаты"""
        for result in results:
            result.relevance_score = RelevanceRanker.calculate_relevance(result, query)

        # Сортировка по релевантности
        ranked = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        return ranked


class MultiSearch:
    """Улучшенный мультипоисковик с параллельными запросами"""

    def __init__(self):
        self.engines = [
            DuckDuckGoSearch(),
            SearxSearch(),
            WikipediaSearch(),
            YandexSearch(),
            GoogleScholarSearch(),
        ]
        self.content_extractor = ContentExtractor()
        self.ranker = RelevanceRanker()
        self.stats = defaultdict(int)

    async def search(self, query: str, deep: bool = False, scholarly: bool = False) -> Dict:
        """
        Выполняет поиск по всем источникам
        Args:
            query: Поисковый запрос
            deep: Извлекать полный контент страниц
            scholarly: Включить поиск научных статей
        """
        # Исправление опечаток
        corrected_query = query
        for typo, correct in TYPO_CORRECTIONS.items():
            corrected_query = corrected_query.replace(typo, correct)

        # Проверка кэша
        cache_key = f"{corrected_query}_{deep}_{scholarly}"
        if cache_key in SEARCH_CACHE:
            cached_data, cached_time = SEARCH_CACHE[cache_key]
            if datetime.now() - cached_time < CACHE_DURATION:
                cached_data['from_cache'] = True
                return cached_data

        print(f"🔍 Поиск: {corrected_query}")
        self.stats['total_searches'] += 1

        # Выбор движков
        active_engines = self.engines.copy()
        if not scholarly:
            # Исключаем Google Scholar для обычного поиска
            active_engines = [e for e in active_engines if e.name != "Google Scholar"]

        # Параллельный поиск
        tasks = [engine.search(corrected_query, limit=5) for engine in active_engines]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Объединение результатов
        all_results = []
        seen_urls = set()
        sources_used = []

        for results in results_list:
            if isinstance(results, list):
                for result in results:
                    url = result.url.strip()
                    # Дедупликация по URL
                    if url and url not in seen_urls and url.startswith('http'):
                        seen_urls.add(url)
                        all_results.append(result)
                        if result.source not in sources_used:
                            sources_used.append(result.source)

        # Ранжирование результатов
        ranked_results = self.ranker.rank_results(all_results, corrected_query)

        # Глубокий поиск (извлечение полного контента)
        if deep and ranked_results:
            extract_tasks = [
                self.content_extractor.extract_from_url(result.url)
                for result in ranked_results[:3]
            ]
            contents = await asyncio.gather(*extract_tasks, return_exceptions=True)

            for result, content in zip(ranked_results[:3], contents):
                if isinstance(content, str) and content:
                    result.full_content = content
                    # Извлечение ключевых фактов
                    key_facts = self.content_extractor.extract_key_facts(content, corrected_query)
                    if key_facts:
                        result.snippet = ' '.join(key_facts[:2])  # Обновляем сниппет

        # Формирование ответа
        response = {
            'query': corrected_query,
            'original_query': query if query != corrected_query else None,
            'results': [r.to_dict() for r in ranked_results[:15]],
            'total_found': len(ranked_results),
            'sources_used': sources_used,
            'timestamp': datetime.now().isoformat(),
            'from_cache': False,
            'deep_search': deep
        }

        # Кэширование
        SEARCH_CACHE[cache_key] = (response, datetime.now())

        # Статистика
        self.stats['successful_searches'] += 1
        self.stats['results_found'] += len(ranked_results)

        return response

    def format_results(self, search_data: Dict, max_results: int = 5) -> str:
        """Форматирует результаты для вывода"""
        output = f"🔍 **Результаты поиска:** {search_data['query']}\n"

        if search_data.get('original_query'):
            output += f"_Исправлено с: {search_data['original_query']}_\n"

        output += f"📊 Найдено: {search_data['total_found']} результатов\n"
        output += f"🌐 Источники: {', '.join(search_data['sources_used'])}\n"

        if search_data.get('from_cache'):
            output += "💾 _Из кэша_\n"

        output += "\n"

        for i, result in enumerate(search_data['results'][:max_results], 1):
            output += f"**{i}. {result['title']}**\n"

            # Релевантность
            if result.get('relevance_score', 0) > 0:
                stars = min(5, int(result['relevance_score'] / 20))
                output += f"⭐ {'★' * stars}{'☆' * (5 - stars)} ({result['relevance_score']:.0f})\n"

            # Сниппет
            snippet = result.get('snippet', '')
            if snippet:
                snippet_preview = snippet[:250] + "..." if len(snippet) > 250 else snippet
                output += f"💬 {snippet_preview}\n"

            output += f"🔗 {result['url']}\n"
            output += f"📌 Источник: {result['source']}\n"

            # Полный контент (если есть)
            if result.get('full_content'):
                content_preview = result['full_content'][:400]
                output += f"\n📄 **Контент:**\n{content_preview}...\n"

            output += "\n"

        return output

    def get_stats(self) -> Dict:
        """Возвращает статистику поиска"""
        engine_stats = {}
        for engine in self.engines:
            engine_stats[engine.name] = {
                'reliability': engine.get_reliability_score(),
                'success': engine.success_count,
                'fails': engine.fail_count
            }

        return {
            'general': dict(self.stats),
            'engines': engine_stats
        }


class SpecializedSearch:
    """Специализированные поисковые запросы"""

    @staticmethod
    async def get_usd_rate() -> Dict:
        """Получает курс USD с ЦБ РФ"""
        try:
            url = "https://www.cbr-xml-daily.ru/daily_json.js"  # ✅ ИСПРАВЛЕНО: убран пробел

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url, ssl=SSL_CONTEXT) as resp:
                    if resp.status == 200:
                        # ✅ ИСПРАВЛЕНО: парсим через text, а не через json()
                        text = await resp.text()
                        data = json.loads(text)

                        usd_data = data['Valute']['USD']
                        return {
                            'rate': round(usd_data['Value'], 2),
                            'previous': round(usd_data['Previous'], 2),
                            'change': round(usd_data['Value'] - usd_data['Previous'], 2),
                            'date': data['Date'].split('T')[0],
                            'source': 'Центральный Банк РФ',
                            'timestamp': datetime.now().isoformat()
                        }
        except Exception as e:
            print(f"Ошибка получения курса USD: {e}")
            return {'error': 'Не удалось получить курс'}

    @staticmethod
    async def get_bitcoin_price() -> Dict:
        """Получает цену Bitcoin"""
        # Пробуем несколько источников
        sources = [
            SpecializedSearch._get_btc_coingecko,
            SpecializedSearch._get_btc_coindesk,
        ]

        for source_func in sources:
            try:
                result = await source_func()
                if 'error' not in result:
                    return result
            except:
                continue

        return {'error': 'Не удалось получить цену Bitcoin'}

    @staticmethod
    async def _get_btc_coingecko() -> Dict:
        """CoinGecko API (бесплатный)"""
        try:
            url = "https://api.coingecko.com/api/v3/simple/price"  # ✅ ИСПРАВЛЕНО: убран пробел
            params = {
                'ids': 'bitcoin',
                'vs_currencies': 'usd,rub',
                'include_24hr_change': 'true'
            }

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url, params=params, ssl=SSL_CONTEXT) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        btc = data['bitcoin']
                        return {
                            'price_usd': btc['usd'],
                            'price_rub': btc.get('rub'),
                            'change_24h': btc.get('usd_24h_change'),
                            'source': 'CoinGecko API',
                            'timestamp': datetime.now().isoformat()
                        }
        except Exception as e:
            raise e

    @staticmethod
    async def _get_btc_coindesk() -> Dict:
        """CoinDesk API"""
        try:
            url = "https://api.coindesk.com/v1/bpi/currentprice.json"  # ✅ ИСПРАВЛЕНО: убран пробел

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url, ssl=SSL_CONTEXT) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            'price_usd': data['bpi']['USD']['rate_float'],
                            'source': 'CoinDesk API',
                            'timestamp': datetime.now().isoformat()
                        }
        except Exception as e:
            raise e

    @staticmethod
    async def get_weather(city: str = "Moscow") -> Dict:
        """Получает погоду для города"""
        try:
            url = f"https://wttr.in/{city}?format=j1"  # ✅ ИСПРАВЛЕНО: убран пробел

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url, ssl=SSL_CONTEXT) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('current_condition'):
                            current = data['current_condition'][0]
                            weather = data['weather'][0] if data.get('weather') else {}
                            return {
                                'city': city,
                                'temperature_c': current.get('temp_C', 'N/A'),
                                'feels_like_c': current.get('FeelsLikeC', 'N/A'),
                                'description': current.get('weatherDesc', [{}])[0].get('value', 'N/A'),
                                'humidity': current.get('humidity', 'N/A'),
                                'wind_speed': current.get('windspeedKmph', 'N/A'),
                                'max_temp': weather.get('maxtempC', 'N/A'),
                                'min_temp': weather.get('mintempC', 'N/A'),
                                'source': 'wttr.in',
                                'timestamp': datetime.now().isoformat()
                            }
        except Exception as e:
            print(f"Ошибка получения погоды: {e}")
            return {'error': 'Не удалось получить погоду'}

    @staticmethod
    async def get_crypto_price(symbol: str) -> Dict:
        """Получает цену любой криптовалюты"""
        try:
            url = "https://api.coingecko.com/api/v3/simple/price"  # ✅ ИСПРАВЛЕНО: убран пробел
            params = {
                'ids': symbol.lower(),
                'vs_currencies': 'usd,rub',
                'include_24hr_change': 'true'
            }

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url, params=params, ssl=SSL_CONTEXT) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if symbol.lower() in data:
                            crypto = data[symbol.lower()]
                            return {
                                'symbol': symbol.upper(),
                                'price_usd': crypto['usd'],
                                'price_rub': crypto.get('rub'),
                                'change_24h': crypto.get('usd_24h_change'),
                                'source': 'CoinGecko',
                                'timestamp': datetime.now().isoformat()
                            }
        except Exception as e:
            print(f"Ошибка получения цены {symbol}: {e}")
            return {'error': f'Не удалось получить цену {symbol}'}


async def test_search():
    """Тестирование всех функций поиска"""
    print("=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ УЛУЧШЕННОГО ПОИСКА v4.1")
    print("=" * 70)

    # Тест 1: Курс доллара
    print("\n1️⃣ ТЕСТ: Курс доллара (ЦБ РФ)")
    print("-" * 70)
    usd = await SpecializedSearch.get_usd_rate()
    if 'error' in usd:
        print(f"❌ {usd['error']}")
    else:
        print(f"💵 Курс USD: {usd['rate']} RUB")
        print(f"📊 Изменение: {usd['change']:+.2f} RUB")
        print(f"📅 Дата: {usd['date']}")

    # Тест 2: Bitcoin
    print("\n2️⃣ ТЕСТ: Курс Bitcoin")
    print("-" * 70)
    btc = await SpecializedSearch.get_bitcoin_price()
    if 'error' in btc:
        print(f"❌ {btc['error']}")
    else:
        print(f"₿ Bitcoin: ${btc['price_usd']:,.2f} USD")
        if btc.get('price_rub'):
            print(f"₿ Bitcoin: {btc['price_rub']:,.2f} RUB")
        if btc.get('change_24h'):
            print(f"📊 Изменение 24ч: {btc['change_24h']:+.2f}%")

    # Тест 3: Погода
    print("\n3️⃣ ТЕСТ: Погода в Москве")
    print("-" * 70)
    weather = await SpecializedSearch.get_weather("Moscow")
    if 'error' in weather:
        print(f"❌ {weather['error']}")
    else:
        print(f"🌤 Температура: {weather['temperature_c']}°C (ощущается как {weather['feels_like_c']}°C)")
        print(f"☁️  {weather['description']}")
        print(f"💧 Влажность: {weather['humidity']}%")
        print(f"💨 Ветер: {weather['wind_speed']} км/ч")

    # Тест 4: Общий поиск
    print("\n4️⃣ ТЕСТ: Общий поиск (без deep)")
    print("-" * 70)
    multi = MultiSearch()
    results = await multi.search("искусственный интеллект 2024", deep=False)
    print(f"✅ Найдено: {results['total_found']} результатов")
    print(f"🌐 Источники: {', '.join(results['sources_used'])}")
    for i, r in enumerate(results['results'][:3], 1):
        print(f"\n{i}. {r['title'][:60]}...")
        print(f"   Релевантность: {r['relevance_score']:.0f}")

    # Тест 5: Глубокий поиск
    print("\n5️⃣ ТЕСТ: Глубокий поиск (с извлечением контента)")
    print("-" * 70)
    deep_results = await multi.search("нейронные сети", deep=True)
    print(f"✅ Найдено: {deep_results['total_found']} результатов")
    for i, r in enumerate(deep_results['results'][:2], 1):
        print(f"\n{i}. {r['title'][:50]}...")
        if r.get('full_content'):
            print(f"   📄 Контент извлечен: {len(r['full_content'])} символов")

    # Статистика
    print("\n" + "=" * 70)
    print("📊 СТАТИСТИКА ПОИСКА")
    print("=" * 70)
    stats = multi.get_stats()
    print(f"Всего поисков: {stats['general']['total_searches']}")
    print(f"Успешных: {stats['general']['successful_searches']}")
    print(f"Результатов найдено: {stats['general']['results_found']}")
    print("\n🔧 Надежность движков:")
    for engine, data in stats['engines'].items():
        reliability_percent = data['reliability'] * 100
        print(f"  {engine}: {reliability_percent:.0f}% (✓{data['success']} ✗{data['fails']})")

    print("\n" + "=" * 70)
    print("✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_search())