# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class FactCheckCore(KnowledgeCore):
    name = "fact_check_core"
    description = "Проверка фактов через веб-поиск и анализ источников"
    capabilities = ["проверка фактов", "поиск подтверждения", "анализ источников"]

    def can_handle(self, query):
        q = query.lower()
        fact_keywords = [
            "факт", "подтвердить", "источник", "проверить", 
            "новости", "курс валют", "событие", "дата"
        ]
        return any(keyword in q for keyword in fact_keywords)

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context or 'web_search' not in context['tools']:
                return {
                    'success': False,
                    'result': "❌ Инструменты недоступны",
                    'data': None,
                    'requires_llm': True
                }

            # Поиск информации по запросу
            search_results = context['tools']['web_search'](query)
            if not search_results:
                return {
                    'success': False,
                    'result': "❌ Не найдено результатов поиска",
                    'data': None,
                    'requires_llm': True
                }

            # Формируем сводку из первых двух источников
            snippets = []
            for result in search_results[:2]:
                if 'snippet' in result:
                    snippets.append(f"- {result['snippet']}")

            summary = "\n".join(snippets) if snippets else "Нет доступных сведений"

            # Проверяем наличие памяти для дополнительного анализа
            memory_data = context.get('memory', {}).get('long_term', {})
            fact_check_info = f"🔍 Поиск: {query}\n\nСводка источников:\n{summary}"

            if memory_data:
                fact_check_info += f"\n\n🧠 Дополнительная информация из памяти:\n{str(memory_data)[:200]}..."

            return {
                'success': True,
                'result': fact_check_info,
                'data': {'sources': search_results[:2]},
                'requires_llm': False
            }

        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при проверке факта: {str(e)}",
                'data': None,
                'requires_llm': True
            }
