# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class CreativeIdeaCore(KnowledgeCore):
    name = "creative_idea_core"
    description = "Генерация идей и предложений для проектов и решений"
    capabilities = ["генерация идей", "творческое решение", "советы по проектам"]

    def can_handle(self, query):
        q = query.lower()
        keywords = ["что придумать", "идея для проекта", "как решить проблему", "новая идея", "творческая задача"]
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

            search_query = f"творческие идеи {query}"
            results = context['tools']['web_search'](search_query)

            if not results:
                return {
                    'success': False,
                    'result': "❌ Не удалось найти подходящие идеи",
                    'data': None,
                    'requires_llm': True
                }

            ideas = []
            for result in results[:3]:
                snippet = result.get('snippet', '')
                if snippet:
                    ideas.append(snippet)

            if not ideas:
                return {
                    'success': False,
                    'result': "❌ Не удалось извлечь идеи из поиска",
                    'data': None,
                    'requires_llm': True
                }

            idea_list = "\n".join([f"• {idea}" for idea in ideas])
            return {
                'success': True,
                'result': f"🧠 Вот несколько идей:\n{idea_list}",
                'data': ideas,
                'requires_llm': False
            }

        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при генерации идей: {str(e)}",
                'data': None,
                'requires_llm': True
            }