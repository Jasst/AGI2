# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class RiskAssessmentCore(KnowledgeCore):
    name = "risk_assessment_core"
    description = "Оценка рисков, анализ последствий и рекомендации по минимизации"
    capabilities = ["оценка рисков", "анализ последствий", "рекомендации по безопасности"]

    def can_handle(self, query):
        q = query.lower()
        keywords = [
            "что может пойти не так",
            "какие риски",
            "как снизить вероятность ошибки",
            "анализ рисков",
            "риск-анализ",
            "последствия ошибки"
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

            search_query = f"анализ рисков и последствий {query}"
            results = context['tools']['web_search'](search_query)

            if not results:
                return {
                    'success': False,
                    'result': "❌ Не удалось найти информацию по запросу",
                    'data': None,
                    'requires_llm': True
                }

            snippets = []
            for r in results[:3]:
                snippets.append(f"- {r['snippet']}")

            snippet_text = '\n'.join(snippets)

            return {
                'success': True,
                'result': f"🔍 Анализ рисков:\n{snippet_text}",
                'data': {
                    'query': query,
                    'snippets': snippets
                },
                'requires_llm': False
            }

        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при оценке рисков: {str(e)}",
                'data': None,
                'requires_llm': True
            }
