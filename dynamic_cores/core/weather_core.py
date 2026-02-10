# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class WeatherCore(KnowledgeCore):
    name = "weather_core"
    description = "Актуальная погода и прогноз через веб-поиск и Open-Meteo API"
    capabilities = ["текущая погода", "прогноз на день", "температура", "влажность", "ветер"]

    def can_handle(self, query):
        q = query.lower()
        return any(kw in q for kw in ['погода', 'температура', 'прогноз', 'дождь'])

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context or 'web_search' not in context['tools']:
                return {
                    'success': False,
                    'result': "❌ Нет доступа к инструментам поиска",
                    'data': None,
                    'requires_llm': True
                }

            search_results = context['tools']['web_search'](query)
            if not search_results:
                return {
                    'success': False,
                    'result': "❌ Не удалось найти информацию о погоде",
                    'data': None,
                    'requires_llm': True
                }

            city = None
            for result in search_results:
                if 'погода' in result['snippet'].lower() and 'в' in result['snippet'].lower():
                    # Простая попытка извлечь город из сниппета
                    snippet = result['snippet'].lower()
                    if 'в ' in snippet:
                        parts = snippet.split('в ')
                        if len(parts) > 1:
                            city = parts[1].split()[0]
                            break

            if not city:
                # Если не удалось определить город, пытаемся использовать запрос
                words = query.lower().split()
                for i, word in enumerate(words):
                    if word in ['в', 'во', 'все']:
                        if i + 1 < len(words):
                            city = words[i + 1]
                            break

            if not city:
                return {
                    'success': False,
                    'result': "❌ Не удалось определить город для запроса",
                    'data': None,
                    'requires_llm': True
                }

            import requests
            url = f"https://api.open-meteo.com/v1/forecast?latitude=55.7558&longitude=37.6176&current_weather=true"
            response = requests.get(url)
            if response.status_code != 200:
                return {
                    'success': False,
                    'result': "❌ Не удалось получить данные о погоде",
                    'data': None,
                    'requires_llm': True
                }

            data = response.json()
            current = data.get('current_weather', {})
            if not current:
                return {
                    'success': False,
                    'result': "❌ Нет данных о текущей погоде",
                    'data': None,
                    'requires_llm': True
                }

            temp = current.get('temperature', 'неизвестно')
            wind_speed = current.get('windspeed', 'неизвестно')
            condition = current.get('weathercode', 'неизвестно')

            # Простая эмуляция условий
            conditions_map = {
                0: "ясно",
                1: "преимущественно ясно",
                2: "облачно",
                3: "пасмурно",
                45: "туман",
                48: "легкий туман",
                51: "слабый дождь",
                53: "дождь",
                55: "сильный дождь"
            }
            condition_text = conditions_map.get(condition, "неизвестно")

            result_str = f"🌤 Погода в {city}: {temp}°C, {condition_text}, ветер {wind_speed} м/с"

            return {
                'success': True,
                'result': result_str,
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
