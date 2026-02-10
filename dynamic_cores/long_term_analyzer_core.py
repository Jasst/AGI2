# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class LongTermAnalyzerCore(KnowledgeCore):
    name = "long_term_analyzer_core"
    description = "Анализ накопленной памяти пользователя для выявления предпочтений и повторяющихся запросов"
    capabilities = ["анализ памяти", "выявление предпочтений", "подготовка рекомендаций"]

    def can_handle(self, query):
        q = query.lower()
        return any(keyword in q for keyword in [
            "предпочтения", "история", "диалоги", "память", "анализ", "рекомендации"
        ])

    def execute(self, query, context=None):
        try:
            if not hasattr(self, 'memory') or not hasattr(self.memory, 'long_term'):
                return {
                    'success': False,
                    'result': "❌ Нет доступа к долгосрочной памяти",
                    'data': None,
                    'requires_llm': True
                }

            long_term_data = self.memory.long_term
            if not long_term_data:
                return {
                    'success': False,
                    'result': "❌ Данные памяти отсутствуют",
                    'data': None,
                    'requires_llm': True
                }

            # Пример анализа: выделяем ключевые темы и частоту запросов
            topics = {}
            preferences = []
            for entry in long_term_data:
                if isinstance(entry, dict) and 'query' in entry:
                    q = entry['query'].lower()
                    if 'предпочтение' in q or 'люблю' in q or 'не люблю' in q:
                        preferences.append(q)
                    # Подсчет тем
                    for word in ['еда', 'музыка', 'кино', 'книги', 'путешествие']:
                        if word in q:
                            topics[word] = topics.get(word, 0) + 1

            summary = f"📊 Сводка по памяти:\n"
            if topics:
                summary += "Темы: " + ", ".join([f"{k} ({v})" for k, v in topics.items()]) + "\n"
            if preferences:
                summary += "Предпочтения: " + "; ".join(preferences[:3]) + "\n"

            recommendations = [
                "Регулярно обновляйте свои предпочтения",
                "Используйте историю для персонализации рекомендаций"
            ]

            return {
                'success': True,
                'result': summary,
                'data': {
                    'topics': topics,
                    'preferences': preferences,
                    'recommendations': recommendations
                },
                'requires_llm': False
            }

        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при анализе памяти: {str(e)}",
                'data': None,
                'requires_llm': True
            }