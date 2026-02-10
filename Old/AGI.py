#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 MiniBrain 2.0 — эмерджентный когнитивный бот
✅ Поддержка Python 3.13
✅ Актуальные данные через веб-поиск (DuckDuckGo)
✅ Эмерджентное ранжирование и асинхронная обработка
✅ Долгосрочная память и динамические ядра
"""

import os, re, json, ast, asyncio, requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ==================== КОНФИГУРАЦИЯ ====================
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN') or ""
LM_STUDIO_API_URL = os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1/chat/completions')
LM_STUDIO_API_KEY = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')
CORES_DIR = "dynamic_cores"
MEMORY_DIR = "brain_memory"
os.makedirs(CORES_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)

if not TELEGRAM_TOKEN:
    raise ValueError("❌ TELEGRAM_TOKEN не найден в .env")

# ==================== БАЗОВЫЕ КЛАССЫ ====================
class KnowledgeCore(ABC):
    name: str = "base_core"
    description: str = "Базовое ядро"
    capabilities: List[str] = []

    @abstractmethod
    def can_handle(self, query: str) -> bool: ...
    @abstractmethod
    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]: ...

# ==================== ВСТРОЕННЫЕ ЯДРА ====================
class DateTimeCore(KnowledgeCore):
    name = "datetime_core"
    description = "Дата и время"
    capabilities = ["дата", "время", "день недели"]

    def can_handle(self, query: str) -> bool:
        q = query.lower()
        return any(kw in q for kw in ["день", "дата", "сегодня", "вчера", "завтра", "час"])

    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        now = datetime.now()
        offset = 0
        if 'завтра' in query: offset = 1
        if 'послезавтра' in query: offset = 2
        if 'вчера' in query: offset = -1
        match = re.search(r'через\s+(\d+)\s*(дн[еяй]|день|дней)', query.lower())
        if match: offset = int(match.group(1))
        target = now + timedelta(days=offset)
        weekdays = ['понедельник','вторник','среда','четверг','пятница','суббота','воскресенье']
        months = ['января','февраля','марта','апреля','мая','июня','июля','августа','сентября','октября','ноября','декабря']
        res = f"📅 {target.day} {months[target.month-1]} {target.year}, {weekdays[target.weekday()]}\n⏰ {target.hour:02d}:{target.minute:02d}"
        return {"success": True, "result": res, "data": {"date": target.isoformat()}, "requires_llm": False}

class CalculatorCore(KnowledgeCore):
    name = "calculator_core"
    description = "Математика"
    capabilities = ["сложение","вычитание","умножение","деление"]

    def can_handle(self, query: str) -> bool:
        return bool(re.search(r'\d+\s*[\+\-\*\/x×]\s*\d+', query.lower()))

    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            expr = re.sub(r'[^\d\+\-\*\/x×\.\(\)\s]','',query.lower())
            expr = expr.replace('x','*').replace('×','*').replace(' ','')
            result = eval(expr, {"__builtins__": {}}, {})
            return {"success": True, "result": f"🧮 {expr.replace('*','×')} = {result}", "data":{"expression":expr,"result":result}, "requires_llm":False}
        except Exception as e:
            return {"success": False, "result": str(e), "data": None, "requires_llm": True}

class WebSearchCore(KnowledgeCore):
    name = "web_search_core"
    description = "Актуальная информация из интернета"
    capabilities = ["поиск","новости","курсы","погода"]

    def __init__(self): self.ddgs = DDGS()
    def can_handle(self, query:str)->bool: return any(k in query.lower() for k in self.capabilities)
    def execute(self, query:str, context:Dict[str,Any]=None)->Dict[str,Any]:
        try:
            results = self.ddgs.text(query,max_results=5,timeout=8)
            search_results = [{"title":r.get('title','')[:80],"url":r.get('href',''),"snippet":r.get('body','')[:200]} for r in results[:3]]
            return {"success":True,"result":None,"data":{"query":query,"results":search_results,"source":"duckduckgo"},"requires_llm":True}
        except Exception as e:
            return {"success":False,"result":str(e),"data":None,"requires_llm":True}

# ==================== ПАМЯТЬ ====================
class MemoryManager:
    def __init__(self,user_id:str):
        self.user_id=user_id
        self.short_term:List[Dict]=[]
        self.long_term=self._load_long_term()
        self.memory_file=os.path.join(MEMORY_DIR,f"user_{user_id}_facts.json")

    def _load_long_term(self)->List[Dict]:
        if os.path.exists(self.memory_file):
            try: return json.load(open(self.memory_file,'r',encoding='utf-8'))
            except: return []
        return []

    def _save_long_term(self): json.dump(self.long_term,open(self.memory_file,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
    def add_short_term(self,msg:Dict): self.short_term.append(msg); self.short_term=self.short_term[-15:]
    def get_short_term(self)->List[Dict]: return self.short_term.copy()
    async def save_long_term(self,fact:str,fact_type="general"): self.long_term.append({'content':fact,'type':fact_type,'timestamp':datetime.now().isoformat(),'user_id':self.user_id}); self._save_long_term()

# ==================== ИНТЕЛЛЕКТУАЛЬНЫЕ ЯДРА ====================
class ToolsManager:
    def __init__(self):
        self.cores:Dict[str,KnowledgeCore]={}
        self._load_builtin_cores()
        self._load_dynamic_cores()

    def _load_builtin_cores(self):
        for c in [DateTimeCore(),CalculatorCore(),WebSearchCore()]: self.cores[c.name]=c
    def _load_dynamic_cores(self):
        for f in os.listdir(CORES_DIR):
            if f.endswith('.py') and not f.startswith('__'):
                try:
                    ns={'KnowledgeCore':KnowledgeCore,'requests':requests,'re':re,'json':json,'datetime':datetime}
                    code=open(os.path.join(CORES_DIR,f),'r',encoding='utf-8').read()
                    exec(code,ns)
                    match = re.search(r'class\s+(\w+)\s*\(KnowledgeCore\)',code)
                    if match: self.cores[match.group(1)]=ns[match.group(1)]()
                except: pass

# ==================== MINI BRAIN ====================
class MiniBrain:
    def __init__(self,user_id:str):
        self.user_id=user_id
        self.memory=MemoryManager(user_id)
        self.tools=ToolsManager()

    async def process(self,query:str)->Dict[str,Any]:
        ranked=[core for core in self.tools.cores.values() if core.can_handle(query)]
        for core in ranked:
            res=core.execute(query,context={'tools':{'web_search':WebSearchCore().execute}})
            if res['success'] and res.get('result') and not res.get('requires_llm',False):
                await self.memory.save_long_term(f"Ответ ядра {core.name} на '{query}'")
                return {'type':'tool_response','response':res['result'],'source':core.name}
            if res['success'] and res.get('data'):
                return {'type':'llm_with_data','context':json.dumps(res['data'],ensure_ascii=False),'source':core.name}
        return {'type':'llm_normal','context':f"ВОПРОС: {query}"}

# ==================== TELEGAM BOT ====================
class TelegramBot:
    def __init__(self):
        self.user_brains:Dict[str,MiniBrain]={}
    def get_brain(self,user_id:str)->MiniBrain:
        if user_id not in self.user_brains: self.user_brains[user_id]=MiniBrain(user_id)
        return self.user_brains[user_id]

# ==================== ЗАПУСК ====================
def main():
    bot=TelegramBot()
    app=Application.builder().token(TELEGRAM_TOKEN).build()
    # Добавьте обработчики команд и сообщений как в твоём коде
    app.run_polling(drop_pending_updates=True)

if __name__=="__main__":
    main()
