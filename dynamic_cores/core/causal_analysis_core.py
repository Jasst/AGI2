# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class CausalAnalysisCore(KnowledgeCore):
    name = "causal_analysis_core"
    description = "Анализ причинно-следственных связей и логических цепочек"
    capabilities = ["определение причин", "анализ следствий", "логические выводы", "прогнозирование"]

    def can_handle(self, query):
        q = query.lower()
        return any(kw in q for kw in ['почему', 'если', 'следствие', 'причина'])

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context or 'web_search' not in context['tools']:
                return {
                    'success': False,
                    'result': "❌ Не доступен инструмент поиска",
                    'data': None,
                    'requires_llm': True
                }

            search_query = f"причины и следствия {query}"
            results = context['tools']['web_search'](search_query)

            if not results:
                return {
                    'success': False,
                    'result': "❌ Нет данных для анализа причинно-следственных связей",
                    'data': None,
                    'requires_llm': True
                }

            # Собираем ключевые фразы из результатов
            snippets = []
            for r in results[:3]:
                if 'snippet' in r:
                    snippets.append(r['snippet'])

            if not snippets:
                return {
                    'success': False,
                    'result': "❌ Не удалось извлечь информацию для анализа",
                    'data': None,
                    'requires_llm': True
                }

            # Формируем структурированный вывод
            summary = "\n".join([f"- {s}" for s in snippets])
            return {
                'success': True,
                'result': f"🔍 Анализ причинно-следственных связей:\n{summary}",
                'data': {
                    'causes': [],
                    'effects': [],
                    'logical_chains': []
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
