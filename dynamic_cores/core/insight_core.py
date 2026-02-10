# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class InsightCore(KnowledgeCore):
    name = "insight_core"
    description = "Глубокий анализ и выявление скрытых закономерностей через веб-поиск и память"
    capabilities = ["генерация инсайтов", "анализ данных", "вывод неожиданных закономерностей"]

    def can_handle(self, query):
        q = query.lower()
        analytic_keywords = [
            "что важно", "какая суть", "какие закономерности", 
            "анализ", "инсайт", "неожиданное", "связь", "вывод"
        ]
        return any(kw in q for kw in analytic_keywords)

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context:
                return {
                    'success': False,
                    'result': "❌ Нет доступных инструментов",
                    'data': None,
                    'requires_llm': True
                }

            search_results = context['tools']['web_search'](query)
            if not search_results:
                return {
                    'success': False,
                    'result': "❌ Не удалось найти данные для анализа",
                    'data': None,
                    'requires_llm': True
                }

            snippets = [r["snippet"] for r in search_results[:3] if "snippet" in r]
            if not snippets:
                return {
                    'success': False,
                    'result': "❌ Нет доступных текстов для анализа",
                    'data': None,
                    'requires_llm': True
                }

            combined_text = "\n".join(snippets)
            insight_prompt = f"Проанализируй следующую информацию и выдай краткий, неожиданный инсайт:\n{combined_text}\n\nИнсайт:"

            return {
                'success': True,
                'result': insight_prompt,
                'data': {'text': combined_text},
                'requires_llm': True
            }
        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при генерации инсайта: {str(e)}",
                'data': None,
                'requires_llm': True
            }
