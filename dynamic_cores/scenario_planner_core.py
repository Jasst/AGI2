# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class ScenarioPlannerCore(KnowledgeCore):
    name = "scenario_planner_core"
    description = "Стратегическое планирование проектов с поэтапными планами и управлением ресурсами"
    capabilities = ["планирование проекта", "управление ресурсами", "контроль сроков"]

    def can_handle(self, query):
        q = query.lower()
        keywords = ['проект', 'задача', 'план', 'реализация', 'организация', 'стратегия', 'цель']
        return any(kw in q for kw in keywords) and ('как реализовать' in q or 'план действий' in q or 'организовать задачу' in q)

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context or 'web_search' not in context['tools']:
                return {
                    'success': False,
                    'result': "❌ Не доступен инструмент поиска",
                    'data': None,
                    'requires_llm': True
                }

            search_query = f"стратегическое планирование проекта {query}"
            results = context['tools']['web_search'](search_query)

            if not results:
                return {
                    'success': False,
                    'result': "❌ Не удалось найти информацию по запросу",
                    'data': None,
                    'requires_llm': True
                }

            # Формируем ответ на основе результатов
            snippets = []
            for r in results[:3]:
                snippets.append(f"- {r['snippet']}")

            plan_text = "\n".join(snippets)

            return {
                'success': True,
                'result': f"📊 Стратегический план для '{query}':\n{plan_text}",
                'data': {
                    'steps': [
                        "1. Определение целей и задач",
                        "2. Анализ ресурсов и ограничений",
                        "3. Разработка поэтапного плана",
                        "4. Контроль сроков и рисков",
                        "5. Оценка прогресса и корректировка"
                    ],
                    'recommendations': [
                        "Используйте методы управления проектами (например, Agile или Waterfall)",
                        "Оценивайте риски на каждом этапе",
                        "Регулярно проводите встречи с командой"
                    ]
                },
                'requires_llm': False
            }

        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при формировании плана: {str(e)}",
                'data': None,
                'requires_llm': True
            }
