# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class AnalyticsCore(KnowledgeCore):
    name = "analytics_core"
    description = "Анализ данных и прогнозирование событий через веб-поиск"
    capabilities = ["анализ финансовых данных", "прогноз погоды", "тренды и статистика", "аналитические отчеты"]

    def can_handle(self, query):
        q = query.lower()
        keywords = ['прогноз', 'анализ', 'тренд', 'событие', 'курс']
        return any(kw in q for kw in keywords)

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context or 'web_search' not in context['tools']:
                return {
                    'success': False,
                    'result': "❌ Инструменты недоступны",
                    'data': None,
                    'requires_llm': True
                }

            results = context['tools']['web_search'](query)
            if not results:
                return {
                    'success': False,
                    'result': "❌ Не найдено данных для анализа",
                    'data': None,
                    'requires_llm': True
                }

            snippets = []
            for result in results[:3]:
                snippets.append(f"- {result.get('snippet', '')}")

            data_summary = '\n'.join(snippets)
            return {
                'success': True,
                'result': f"📊 Результаты анализа:\n{data_summary}",
                'data': {
                    'query': query,
                    'sources': [r.get('url', '') for r in results[:3]]
                },
                'requires_llm': False
            }
        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при анализе: {str(e)}",
                'data': None,
                'requires_llm': True
            }
