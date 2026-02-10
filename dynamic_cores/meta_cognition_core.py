# AUTO-GENERATED CORE - created by MiniBrain
# FIXED: No type annotations to avoid 'Dict not defined' error
from __main__ import KnowledgeCore, requests, re, json, datetime, os

class MetaCognitionCore(KnowledgeCore):
    name = "meta_cognition_core"
    description = "Метакогнитивный анализ ответов ИИ: оценка уверенности, поиск противоречий и точности"
    capabilities = ["анализ уверенности", "поиск противоречий", "оценка точности"]

    def can_handle(self, query):
        q = query.lower()
        keywords = [
            "уверенность", "противоречие", "точность", "проверь", "анализ",
            "оценка", "надежность", "подтверждение", "неопределенность"
        ]
        return any(kw in q for kw in keywords)

    def execute(self, query, context=None):
        try:
            if not context or 'tools' not in context:
                return {
                    'success': False,
                    'result': "❌ Нет доступа к инструментам",
                    'data': None,
                    'requires_llm': True
                }

            search_results = context['tools']['web_search'](query)
            if not search_results:
                return {
                    'success': False,
                    'result': "❌ Не удалось выполнить поиск для анализа",
                    'data': None,
                    'requires_llm': True
                }

            snippets = [r.get('snippet', '') for r in search_results[:3] if r.get('snippet')]
            if not snippets:
                return {
                    'success': False,
                    'result': "❌ Нет текстовых данных для анализа",
                    'data': None,
                    'requires_llm': True
                }

            # Пример анализа: проверка на противоречия и неопределенность
            text_content = '\n'.join(snippets)
            confidence_score = self._assess_confidence(text_content)
            contradiction_indicators = self._detect_contradictions(text_content)

            analysis_result = {
                'confidence': confidence_score,
                'contradictions': contradiction_indicators,
                'recommendations': self._generate_recommendations(confidence_score, contradiction_indicators)
            }

            return {
                'success': True,
                'result': f"📊 Метакогнитический анализ:\n{analysis_result['recommendations']}",
                'data': analysis_result,
                'requires_llm': False
            }
        except Exception as e:
            return {
                'success': False,
                'result': f"❌ Ошибка при анализе: {str(e)}",
                'data': None,
                'requires_llm': True
            }

    def _assess_confidence(self, text):
        if not text:
            return 0.0
        words = text.split()
        if len(words) < 10:
            return 0.3
        confidence_indicators = [
            "определенно", "точно", "ясно", "наверняка", "уверен", "доказательство",
            "подтверждено", "доказано", "предсказуемо"
        ]
        score = sum(1 for word in confidence_indicators if word in text.lower())
        return min(score / 5.0, 1.0)

    def _detect_contradictions(self, text):
        contradictions = []
        if "не" in text and ("все" in text or "никогда" in text):
            contradictions.append("Противоречие между утверждением и отрицанием")
        if "возможно" in text.lower() and "обязательно" in text.lower():
            contradictions.append("Несовместимость вероятностного и детерминированного утверждения")
        return contradictions

    def _generate_recommendations(self, confidence, contradictions):
        recs = []
        if confidence < 0.5:
            recs.append("⚠️ Низкая уверенность в ответе — рекомендуется дополнительная проверка")
        if contradictions:
            recs.append("❌ Обнаружены противоречия — требуется уточнение информации")
        if not recs:
            recs.append("✅ Ответ выглядит согласованным и достоверным")
        return "\n".join(recs)
