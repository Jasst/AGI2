# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class KnowledgeAggregatorCore(KnowledgeCore):
    name = "knowledge_aggregator_core"
    description = "Объединяет информацию из разных источников для создания сводных отчётов"
    capabilities = ["объединение данных", "сводка информации", "подготовка отчётов"]

    def can_handle(self, query):
        q = query.lower()
        return any(keyword in q for keyword in [
            "объедини", "сводка", "отчёт", "информация", "данные", "анализ"
        ])

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context:
                return {
                    'success': False,
                    'result': "❌ Нет доступа к инструментам",
                    'data': None,
                    'requires_llm': True
                }

            # Поиск в интернете
            search_results = []
            if 'web_search' in context['tools']:
                search_results = context['tools']['web_search'](query)

            # Извлечение текстов из результатов поиска
            snippets = []
            if search_results:
                for result in search_results[:3]:
                    if 'snippet' in result:
                        snippets.append(result['snippet'])

            # Если есть данные из памяти или других ядер
            aggregated_data = {
                'query': query,
                'web_snippets': snippets,
                'from_memory': context.get('long_term', {}).get('knowledge', []),
                'from_other_cores': context.get('other_cores_data', [])
            }

            # Формируем результат
            result_text = f"📊 Сводка по запросу: {query}\n\n"
            if snippets:
                result_text += "🔍 Результаты поиска:\n"
                for i, snippet in enumerate(snippets, 1):
                    result_text += f"{i}. {snippet}\n\n"

            if 'long_term' in context and 'knowledge' in context['long_term']:
                result_text += "💾 Данные из памяти:\n"
                for item in context['long_term']['knowledge'][:3]:
                    result_text += f"- {item}\n"

            return {
                'success': True,
                'result': result_text,
                'data': aggregated_data,
                'requires_llm': False
            }

        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при объединении данных: {str(e)}",
                'data': None,
                'requires_llm': True
            }
