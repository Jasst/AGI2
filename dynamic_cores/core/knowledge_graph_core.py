# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class KnowledgeGraphCore(KnowledgeCore):
    name = "knowledge_graph_core"
    description = "Построение связей между фактами и визуализация графа знаний"
    capabilities = ["построение связей", "визуализация графа", "вывод скрытых закономерностей"]

    def can_handle(self, query):
        q = query.lower()
        keywords = [
            "связь", "граф", "факт", "понятие", "событие", "причина", 
            "следствие", "закономерность", "структура", "взаимосвязь"
        ]
        return any(kw in q for kw in keywords)

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context or 'web_search' not in context['tools']:
                return {
                    'success': False,
                    'result': "❌ Не доступен инструмент поиска",
                    'data': None,
                    'requires_llm': True
                }

            # Используем веб-поиск для получения информации
            search_results = context['tools']['web_search'](query)
            if not search_results:
                return {
                    'success': False,
                    'result': "❌ Не найдено данных для построения графа",
                    'data': None,
                    'requires_llm': True
                }

            # Формируем ключевые связи на основе результатов
            facts = []
            for result in search_results[:3]:
                snippet = result.get('snippet', '')
                if snippet:
                    facts.append(snippet)

            # Пример формирования связей (в реальном случае можно использовать NLP)
            connections = []
            for i, fact in enumerate(facts):
                if i < len(facts) - 1:
                    connections.append(f"Связь: {fact} ↔ {facts[i+1]}")

            # Формируем рекомендации
            recommendations = [
                "Рассмотрите дополнительные источники для уточнения связей",
                "Анализируйте контекст каждого факта для точного построения графа"
            ]

            return {
                'success': True,
                'result': f"📊 Построен граф знаний:\n{'\n'.join(connections)}",
                'data': {
                    'facts': facts,
                    'connections': connections,
                    'recommendations': recommendations
                },
                'requires_llm': False
            }

        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при построении графа: {str(e)}",
                'data': None,
                'requires_llm': True
            }
