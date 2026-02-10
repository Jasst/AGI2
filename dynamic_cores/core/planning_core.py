# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class PlanningCore(KnowledgeCore):
    name = "planning_core"
    description = "Генерация пошаговых планов и стратегий решения задач"
    capabilities = ["пошаговые инструкции", "планирование задач", "контроль прогресса"]

    def can_handle(self, query):
        q = query.lower()
        return any(kw in q for kw in ['как сделать', 'план действий', 'организация проекта', 'планирование'])

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context or 'web_search' not in context['tools']:
                return {
                    'success': False,
                    'result': "❌ Не доступен инструмент поиска",
                    'data': None,
                    'requires_llm': True
                }

            search_query = f"как создать план действий для: {query}"
            results = context['tools']['web_search'](search_query)

            if not results:
                return {
                    'success': False,
                    'result': "❌ Не удалось найти информацию для планирования",
                    'data': None,
                    'requires_llm': True
                }

            plan_snippets = []
            for result in results[:3]:
                plan_snippets.append(f"- {result['snippet']}")

            plan_text = "\n".join(plan_snippets)

            return {
                'success': True,
                'result': f"📋 План действий:\n{plan_text}",
                'data': None,
                'requires_llm': False
            }

        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при генерации плана: {str(e)}",
                'data': None,
                'requires_llm': True
            }
