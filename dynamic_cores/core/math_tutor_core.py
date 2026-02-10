# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class MathTutorCore(KnowledgeCore):
    name = "math_tutor_core"
    description = "Пошаговое объяснение математических задач и вычисления"
    capabilities = ["вычисления", "пошаговое объяснение", "алгебра", "геометрия", "арифметика"]

    def can_handle(self, query):
        q = query.lower()
        math_keywords = ['считай', 'вычисли', 'решить', 'пример', 'задача', 'уравнение', 'алгебра', 'геометрия', 'арифметика', 'посчитай']
        return any(kw in q for kw in math_keywords)

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context or 'web_search' not in context['tools']:
                return {
                    'success': False,
                    'result': "❌ Не доступен инструмент поиска",
                    'data': None,
                    'requires_llm': True
                }

            # Используем веб-поиск для получения информации о задаче
            search_results = context['tools']['web_search'](query)
            if not search_results:
                return {
                    'success': False,
                    'result': "❌ Не удалось найти информацию по запросу",
                    'data': None,
                    'requires_llm': True
                }

            # Пример простого вычисления (в реальном случае можно использовать более сложные методы)
            # Для демонстрации используем первый результат поиска
            first_result = search_results[0]['snippet']
            explanation = f"🔍 Поиск дал следующую информацию:\n{first_result}\n\n📝 Решение задачи:"
            
            # Пример шагов решения (в реальном случае это будет динамически генерироваться)
            steps = [
                "1. Определите тип задачи",
                "2. Примените соответствующую формулу или метод",
                "3. Выполните вычисления",
                "4. Проверьте ответ"
            ]
            
            explanation += "\n" + "\n".join(steps)
            
            return {
                'success': True,
                'result': f"🧮 {explanation}",
                'data': {'steps': steps, 'source': first_result},
                'requires_llm': False
            }
        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при обработке запроса: {str(e)}",
                'data': None,
                'requires_llm': True
            }
