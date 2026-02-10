# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class CryptoCore(KnowledgeCore):
    name = "crypto_core"
    description = "Отслеживание криптовалют и финансовых инструментов через веб-поиск и публичные API"
    capabilities = [
        "курс биткоина",
        "курс эфира",
        "топ криптовалют",
        "сравнение курса",
        "анализ рынка"
    ]

    def can_handle(self, query):
        q = query.lower()
        keywords = ['биткоин', 'биток', 'btc', 'эфир', 'ethereum', 'eth', 'криптовалют', 'курс', 'топ', 'рынок']
        return any(kw in q for kw in keywords)

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context or 'web_search' not in context['tools']:
                return {
                    'success': False,
                    'result': "❌ Не доступен инструмент поиска",
                    'data': None,
                    'requires_llm': True
                }

            # Поиск актуальных данных через веб-поиск
            search_results = context['tools']['web_search'](query)
            if search_results:
                snippets = '\n'.join([f'- {r["snippet"]}' for r in search_results[:3]])
                return {
                    'success': True,
                    'result': f"🔍 Поиск по запросу:\n{snippets}",
                    'data': None,
                    'requires_llm': False
                }

            # Если веб-поиск не дал результата, используем CoinGecko API
            import requests

            # Основные криптовалюты для запроса
            coins = "bitcoin,ethereum,cardano,solana,polkadot"
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coins}&vs_currencies=usd&include_24hr_change=true"

            response = requests.get(url)
            if response.status_code != 200:
                return {
                    'success': False,
                    'result': "❌ Не удалось получить данные с CoinGecko",
                    'data': None,
                    'requires_llm': True
                }

            data = response.json()
            if not data:
                return {
                    'success': False,
                    'result': "❌ Нет данных о криптовалютах",
                    'data': None,
                    'requires_llm': True
                }

            # Формируем ответ
            result_text = "📊 Актуальные курсы криптовалют:\n"
            for coin_id, info in data.items():
                if coin_id == "bitcoin":
                    name = "Биткоин"
                elif coin_id == "ethereum":
                    name = "Эфириум"
                elif coin_id == "cardano":
                    name = "Cardano"
                elif coin_id == "solana":
                    name = "Solana"
                elif coin_id == "polkadot":
                    name = "Polkadot"
                else:
                    name = coin_id.capitalize()

                price = info.get("usd", 0)
                change = info.get("usd_24h_change", 0)

                result_text += f"{name}: ${price:,.2f} ({change:+.2f}%)\n"

            return {
                'success': True,
                'result': result_text,
                'data': data,
                'requires_llm': False
            }

        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка: {str(e)}",
                'data': None,
                'requires_llm': True
            }
