# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class LearningCore(KnowledgeCore):
    name = "learning_core"
    description = "Ядро для постоянного обучения и расширения знаний на основе анализа ошибок и новых данных"
    capabilities = ["анализ ошибок", "обучение на основе данных", "улучшение стратегий", "обновление знаний"]

    def can_handle(self, query):
        q = query.lower()
        keywords = ['обуч', 'лучше', 'ошибка', 'анализ', 'стратегия', 'умею', 'улучш', 'знания', 'память', 'предыдущий']
        return any(kw in q for kw in keywords)

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context or 'web_search' not in context['tools']:
                return {
                    'success': False,
                    'result': "❌ Нет доступа к инструментам",
                    'data': None,
                    'requires_llm': True
                }

            # Используем веб-поиск для получения актуальной информации
            search_results = context['tools']['web_search'](query)
            if not search_results:
                return {
                    'success': False,
                    'result': "❌ Не удалось найти информацию по запросу",
                    'data': None,
                    'requires_llm': True
                }

            # Формируем ответ на основе результатов
            snippets = '\n'.join([f'- {r["snippet"]}' for r in search_results[:3]])
            return {
                'success': True,
                'result': f"🧠 Анализ запроса: {query}\n\n💡 Новые данные:\n{snippets}",
                'data': {
                    'query': query,
                    'search_results': snippets
                },
                'requires_llm': False
            }
        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при выполнении: {str(e)}",
                'data': None,
                'requires_llm': True
            }
