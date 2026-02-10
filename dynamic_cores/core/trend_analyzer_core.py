# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class TrendAnalyzerCore(KnowledgeCore):
    name = "trend_analyzer_core"
    description = "Поиск и анализ новых трендов в интернете и социальных сетях"
    capabilities = ["анализ трендов", "поиск новинок", "оценка популярности"]

    def can_handle(self, query):
        q = query.lower()
        keywords = ['тренды', 'популярно', 'новые', 'тренд', 'новинки', 'технологии', 'популярность']
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

            search_query = f"новые тренды {query}"
            results = context['tools']['web_search'](search_query)
            
            if not results:
                return {
                    'success': False,
                    'result': "❌ Не удалось найти информацию о трендах",
                    'data': None,
                    'requires_llm': True
                }

            trends = []
            for result in results[:3]:
                snippet = result.get('snippet', '')
                title = result.get('title', '')
                trends.append(f"{title}: {snippet}")

            trend_list = '\n'.join(trends)
            return {
                'success': True,
                'result': f"🔍 Новые тренды:\n{trend_list}",
                'data': None,
                'requires_llm': False
            }
        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при анализе трендов: {str(e)}",
                'data': None,
                'requires_llm': True
            }
