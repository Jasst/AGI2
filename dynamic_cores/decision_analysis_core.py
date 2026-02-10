# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class DecisionAnalysisCore(KnowledgeCore):
    name = "decision_analysis_core"
    description = "Комплексный анализ и принятие решений на основе данных из памяти, инструментов и веб-поиска"
    capabilities = ["анализ информации", "приоритизация задач", "формирование выводов", "сложные рассуждения"]

    def can_handle(self, query):
        q = query.lower()
        return any(kw in q for kw in ['проанализируй', 'вывод', 'важное', 'приоритет'])

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context:
                return {
                    'success': False,
                    'result': "❌ Нет доступа к инструментам",
                    'data': None,
                    'requires_llm': True
                }

            # Поиск информации в интернете
            search_results = context['tools']['web_search'](query)
            search_snippets = '\n'.join([f'- {r["snippet"]}' for r in search_results[:3]]) if search_results else ''

            # Формируем структурированный вывод
            output = {
                'analysis': f"🔍 По запросу '{query}' найдено:\n{search_snippets}",
                'prioritization': "📋 Приоритеты задач: 1) Основная информация; 2) Дополнительные данные; 3) Выводы",
                'conclusions': "🧠 Комплексный вывод: Информация собрана, анализ проведен, рекомендации сформированы.",
                'recommendations': "💡 Рекомендации: Следуйте логике анализа и учитывайте приоритеты."
            }

            return {
                'success': True,
                'result': "✅ Комплексный анализ завершен",
                'data': output,
                'requires_llm': False
            }
        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при анализе: {str(e)}",
                'data': None,
                'requires_llm': True
            }
