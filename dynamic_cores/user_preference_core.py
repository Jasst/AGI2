# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class UserPreferenceCore(KnowledgeCore):
    name = "user_preference_core"
    description = "Анализирует предпочтения и интересы пользователя для формирования персонализированных рекомендаций"
    capabilities = ["анализ предпочтений", "рекомендации под пользователя", "выявление интересов"]

    def can_handle(self, query):
        q = query.lower()
        keywords = ['любим', 'предпочита', 'интерес', 'хобби', 'предпочтения', 'совет', 'рекомендация']
        return any(kw in q for kw in keywords)

    def execute(self, query, context=None):
        try:
            # Проверяем наличие памяти
            if not hasattr(self, 'memory'):
                return {
                    'success': False,
                    'result': "❌ Не удалось получить данные о предпочтениях пользователя",
                    'data': None,
                    'requires_llm': True
                }

            # Используем встроенный инструмент поиска для дополнительной информации
            search_results = None
            if context and 'tools' in context and 'web_search' in context['tools']:
                search_results = context['tools']['web_search'](query)

            # Формируем ответ на основе памяти и поиска
            long_term = self.memory.long_term if hasattr(self.memory, 'long_term') else {}
            short_term = self.memory.short_term if hasattr(self.memory, 'short_term') else {}

            interests = []
            if long_term:
                interests.extend([k for k in long_term.keys() if k != 'user_profile'])
            if short_term:
                interests.extend([k for k in short_term.keys() if k != 'user_profile'])

            # Формируем рекомендации
            recommendations = []
            if interests:
                recommendations.append("На основе ваших интересов:")
                for interest in interests[:3]:  # ограничиваем до 3 интересов
                    recommendations.append(f"- {interest}")
                if search_results:
                    recommendations.append("Дополнительная информация:")
                    for r in search_results[:2]:
                        recommendations.append(f"  • {r.get('title', 'Без названия')}: {r.get('snippet', '')}")

            result_text = "\n".join(recommendations) if recommendations else "Нет доступных данных для рекомендаций."

            return {
                'success': True,
                'result': result_text,
                'data': {
                    'interests': interests,
                    'recommendations': recommendations
                },
                'requires_llm': False
            }

        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при анализе предпочтений: {str(e)}",
                'data': None,
                'requires_llm': True
            }
