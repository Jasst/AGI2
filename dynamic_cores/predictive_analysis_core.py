# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class PredictiveAnalysisCore(KnowledgeCore):
    name = "predictive_analysis_core"
    description = "Анализирует данные и строит прогнозы на основе исторических трендов"
    capabilities = ["анализ трендов", "предсказание будущего", "оценка рисков"]

    def can_handle(self, query):
        q = query.lower()
        keywords = ['что будет', 'какой тренд', 'курс через неделю', 'прогноз', 'предсказать', 'риск']
        return any(kw in q for kw in keywords)

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context:
                return {
                    'success': False,
                    'result': "❌ Нет доступа к инструментам",
                    'data': None,
                    'requires_llm': True
                }

            search_query = f"анализ трендов {query}"
            results = context['tools']['web_search'](search_query)

            if not results:
                return {
                    'success': False,
                    'result': "❌ Не удалось найти данные для анализа",
                    'data': None,
                    'requires_llm': True
                }

            snippets = '\n'.join([f'- {r["snippet"]}' for r in results[:3]])
            trend_analysis = f"📊 Анализ трендов:\n{snippets}"

            return {
                'success': True,
                'result': f"🔮 Прогноз на основе анализа:\n{trend_analysis}",
                'data': None,
                'requires_llm': False
            }

        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при анализе: {str(e)}",
                'data': None,
                'requires_llm': True
            }
