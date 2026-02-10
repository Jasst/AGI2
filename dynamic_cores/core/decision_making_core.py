# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class DecisionMakingCore(KnowledgeCore):
    name = "decision_making_core"
    description = "Анализ вариантов и принятие обоснованных решений на основе данных из памяти и веб-поиска"
    capabilities = ["анализ вариантов", "принятие решения", "взвешивание плюсов и минусов"]

    def can_handle(self, query):
        q = query.lower()
        return any(kw in q for kw in ['что выбрать', 'какой вариант лучше', 'как решить дилемму', 'анализировать', 'сравнить'])

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context:
                return {
                    'success': False,
                    'result': "❌ Нет доступа к инструментам",
                    'data': None,
                    'requires_llm': True
                }

            search_results = context['tools']['web_search'](query)
            if not search_results:
                return {
                    'success': False,
                    'result': "❌ Не удалось получить данные для анализа",
                    'data': None,
                    'requires_llm': True
                }

            snippets = '\n'.join([f'- {r["snippet"]}' for r in search_results[:3]])
            return {
                'success': True,
                'result': f"🔍 Анализ запроса:\n{snippets}\n\n💡 Рекомендация: рассмотрите варианты с учётом их преимуществ и недостатков.",
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
