# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class ScenarioCore(KnowledgeCore):
    name = "scenario_core"
    description = "Моделирование ситуаций, симуляция сценариев и планирование действий"
    capabilities = ["моделирование ситуаций", "симуляция сценариев", "планирование действий", "прогнозирование результатов"]

    def can_handle(self, query):
        q = query.lower()
        return any(kw in q for kw in ['смоделируй', 'спланируй', 'сценарий', 'варианты развития'])

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context or 'web_search' not in context['tools']:
                return {
                    'success': False,
                    'result': "❌ Не доступен инструмент поиска",
                    'data': None,
                    'requires_llm': True
                }

            search_query = f"модель ситуации {query}"
            results = context['tools']['web_search'](search_query)

            if not results:
                return {
                    'success': False,
                    'result': "❌ Не удалось найти информацию для моделирования",
                    'data': None,
                    'requires_llm': True
                }

            snippets = []
            for r in results[:3]:
                snippets.append(f"- {r['snippet']}")

            snippet_text = '\n'.join(snippets)

            return {
                'success': True,
                'result': f"📊 Модель ситуации:\n{snippet_text}",
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
