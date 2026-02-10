# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class StrategyCore(KnowledgeCore):
    name = "strategy_core"
    description = "Анализ стратегий и рекомендации по действиям"
    capabilities = ["анализ вариантов", "поиск оптимальных стратегий", "рекомендации по действиям"]

    def can_handle(self, query):
        q = query.lower()
        keywords = ['как действовать', 'что выбрать', 'как выиграть', 'стратегия', 'анализ', 'рекомендация']
        return any(kw in q for kw in keywords)

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context or 'web_search' not in context['tools']:
                return {
                    'success': False,
                    'result': "❌ Нет доступа к инструментам поиска",
                    'data': None,
                    'requires_llm': True
                }

            search_query = f"стратегии для {query}"
            results = context['tools']['web_search'](search_query)

            if not results:
                return {
                    'success': False,
                    'result': "❌ Не удалось найти информацию по стратегиям",
                    'data': None,
                    'requires_llm': True
                }

            snippets = '\n'.join([f'- {r["snippet"]}' for r in results[:3]])
            return {
                'success': True,
                'result': f"📊 Рекомендации по стратегии:\n{snippets}",
                'data': None,
                'requires_llm': False
            }
        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при анализе стратегии: {str(e)}",
                'data': None,
                'requires_llm': True
            }