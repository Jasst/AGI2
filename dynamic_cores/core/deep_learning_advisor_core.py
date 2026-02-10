# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class DeepLearningAdvisorCore(KnowledgeCore):
    name = "deep_learning_advisor_core"
    description = "Советы по обучению и саморазвитию через анализ долгосрочной памяти пользователя"
    capabilities = ["план обучения", "советы по саморазвитию", "улучшение навыков"]

    def can_handle(self, query):
        q = query.lower()
        keywords = ['учиться', 'навык', 'память', 'продуктивность', 'саморазвитие', 'улучшить', 'лучше']
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

            search_query = f"советы по саморазвитию и улучшению навыков {query}"
            results = context['tools']['web_search'](search_query)

            if not results:
                return {
                    'success': False,
                    'result': "❌ Не удалось найти информацию",
                    'data': None,
                    'requires_llm': True
                }

            snippets = []
            for r in results[:3]:
                snippets.append(f"- {r['snippet']}")

            response_text = "\n".join(snippets)

            return {
                'success': True,
                'result': f"🧠 Советы по саморазвитию:\n{response_text}",
                'data': None,
                'requires_llm': False
            }

        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при обработке запроса: {str(e)}",
                'data': None,
                'requires_llm': True
            }
