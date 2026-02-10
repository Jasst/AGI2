# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class SimulationCore(KnowledgeCore):
    name = "simulation_core"
    description = "Моделирование гипотетических сценариев и анализ последствий изменений условий"
    capabilities = ["моделирование сценариев", "анализ альтернатив", "что-если анализ"]

    def can_handle(self, query):
        q = query.lower()
        return any(kw in q for kw in ['что если', 'какой результат', 'при изменении', 'если бы'])

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context or 'web_search' not in context['tools']:
                return {
                    'success': False,
                    'result': "❌ Не доступны инструменты поиска",
                    'data': None,
                    'requires_llm': True
                }

            search_query = f"анализ последствий {query}"
            results = context['tools']['web_search'](search_query)

            if not results:
                return {
                    'success': False,
                    'result': "❌ Не найдены данные для моделирования сценариев",
                    'data': None,
                    'requires_llm': True
                }

            snippets = []
            for r in results[:3]:
                snippets.append(f"- {r['snippet']}")

            snippet_text = '\n'.join(snippets)

            return {
                'success': True,
                'result': f"🔍 Моделирование сценария:\n{snippet_text}",
                'data': None,
                'requires_llm': False
            }

        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при моделировании: {str(e)}",
                'data': None,
                'requires_llm': True
            }
