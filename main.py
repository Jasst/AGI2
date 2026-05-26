"""
🌐 main.py — FastAPI server: routes, WebSocket, lifespan
"""

import json
import time
import asyncio
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional, Set
from collections import defaultdict
from datetime import datetime, timedelta
import re

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from config import CONFIG
from logging_setup import setup_logging
from models.teacher import TeacherLLM
from agent import AdvancedAutonomousAgent
import socket
import asyncio

load_dotenv()

logger = setup_logging(CONFIG.base_dir)


# ════════════════════════════════════════════════════════════════
# 🛡️ SECURITY MIDDLEWARE (Windows compatible)
# ════════════════════════════════════════════════════════════════

MALICIOUS_PATTERNS = re.compile(
    r'(shell|system|exec|cmd|wget|curl|bash|powershell|\.\.\/|%2e%2e%2f|passwd|shadow|etc/passwd|bin/bash|cmd\.exe)',
    re.IGNORECASE
)

request_counts = defaultdict(list)
BANNED_IPS = set()
BAN_FILE = CONFIG.base_dir / 'banned_ips.txt'

def load_banned_ips():
    if BAN_FILE.exists():
        with open(BAN_FILE, 'r') as f:
            for line in f:
                BANNED_IPS.add(line.strip())

def save_banned_ip(ip: str):
    with open(BAN_FILE, 'a') as f:
        f.write(f"{ip}\n")

def is_rate_limited(client_ip: str, limit: int = 30, window_seconds: int = 60) -> bool:
    now = datetime.now()
    request_counts[client_ip] = [t for t in request_counts[client_ip] if now - t < timedelta(seconds=window_seconds)]
    if len(request_counts[client_ip]) >= limit:
        return True
    request_counts[client_ip].append(now)
    return False

load_banned_ips()

# ════════════════════════════════════════════════════════════════
# 🌐 FASTAPI APP
# ════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global teacher
    teacher = TeacherLLM(CONFIG.lm_studio_url, CONFIG.lm_studio_key)
    await teacher.connect()
    logger.info(f"🚀 Server started — {CONFIG.host}:{CONFIG.port} [{CONFIG.device.upper()}]")
    print(f"""
╔══════════════════════════════════════════════════╗
║  ⚡ ADVANCED AUTONOMOUS AGENT v4.0               ║
║     Monochrome UI · Mobile First · BlockCoin.ru  ║
╠══════════════════════════════════════════════════╣
║  Device  : {CONFIG.device.upper():<38}║
║  Model   : {CONFIG.n_layers}L-{CONFIG.d_model}D-{CONFIG.n_heads}H{'':<28}║
║  Server  : http://{CONFIG.host}:{CONFIG.port:<27}║
╚══════════════════════════════════════════════════╝
""")
    yield
    for agent in agents.values():
        agent._save_state()
    if teacher:
        await teacher.close()
    logger.info("👋 Server shutdown")

app = FastAPI(title="Advanced AI Agent v4", version="4.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Security middleware
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    client_ip = request.client.host

    if client_ip in BANNED_IPS:
        logger.warning(f"Blocked banned IP: {client_ip}")
        return JSONResponse(status_code=403, content={"error": "Access denied"})

    if is_rate_limited(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        BANNED_IPS.add(client_ip)
        save_banned_ip(client_ip)
        return JSONResponse(status_code=429, content={"error": "Too many requests"})

    url_path = str(request.url.path)
    query = str(request.url.query)

    if MALICIOUS_PATTERNS.search(url_path) or MALICIOUS_PATTERNS.search(query):
        logger.warning(f"🚫 Malicious request blocked from {client_ip}: {request.url}")
        BANNED_IPS.add(client_ip)
        save_banned_ip(client_ip)
        return JSONResponse(status_code=403, content={"error": "Forbidden"})

    return await call_next(request)

# Global state
teacher: Optional[TeacherLLM] = None
agents: Dict[str, AdvancedAutonomousAgent] = {}
ws_connections: Dict[str, Set[WebSocket]] = {}

STATIC_DIR = Path(__file__).parent / "static"

# ════════════════════════════════════════════════════════════════
# 📄 HTML PAGE (load from static/index.html)
# ════════════════════════════════════════════════════════════════

def get_html() -> str:
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    # Fallback if static file missing
    return create_fallback_html()

def create_fallback_html() -> str:
    """Минимальный fallback если index.html не найден"""
    return """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Agent v4</title><style>body{background:#0a0a0a;color:#fff;font-family:sans-serif;text-align:center;padding:2rem}</style>
</head><body><h1>🤖 AI Agent v4.0</h1><p>Загрузите <code>static/index.html</code> для интерфейса</p>
<div id="status"></div><script>const ws=new WebSocket(`ws://${location.host}/ws`);
ws.onmessage=e=>{const d=JSON.parse(e.data);if(d.type==='response')document.getElementById('status').innerHTML+=`<p>🤖: ${d.content}</p>`};
function send(){const i=document.getElementById('inp');ws.send(JSON.stringify({type:'message',content:i.value}));i.value='';}
document.write('<input id="inp"><button onclick="send()">Send</button>');</script></body></html>"""

# ════════════════════════════════════════════════════════════════
# 🚀 ROUTES
# ════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(get_html())

@app.get("/health")
async def health():
    return {"status": "ok", "device": CONFIG.device, "model": f"{CONFIG.n_layers}L-{CONFIG.d_model}D", "active_agents": len(agents)}

@app.get("/status/{user_id}")
async def get_status(user_id: str):
    if user_id in agents:
        return agents[user_id].get_status()
    return {"error": f"Agent '{user_id}' not found"}

@app.post("/reset/{user_id}")
async def reset_agent(user_id: str):
    if user_id in agents:
        agents[user_id]._save_state()
        del agents[user_id]
    return {"status": "reset", "user_id": user_id}

@app.post("/ban/{ip}")
async def ban_ip_manual(ip: str):
    BANNED_IPS.add(ip)
    save_banned_ip(ip)
    return {"status": "banned", "ip": ip}

# ════════════════════════════════════════════════════════════════
# 🔌 WEBSOCKET
# ════════════════════════════════════════════════════════════════

MAX_MESSAGE_SIZE = 10 * 1024  # 10KB

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    client_ip = websocket.client.host
    if client_ip in BANNED_IPS:
        await websocket.close(code=1008, reason="Banned")
        return

    user_id = websocket.query_params.get("user_id", f"user_{int(time.time())}")
    user_id = re.sub(r'[^a-zA-Z0-9_]', '_', user_id)[:64]

    if user_id not in agents:
        agents[user_id] = AdvancedAutonomousAgent(user_id, teacher)

    ws_connections.setdefault(user_id, set()).add(websocket)

    try:
        await websocket.send_json({"type": "status", "metadata": agents[user_id].get_status()})

        while True:
            raw = await websocket.receive_text()

            if len(raw) > MAX_MESSAGE_SIZE:
                await websocket.send_json({"type": "error", "content": "Message too large"})
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "content": "Invalid JSON"})
                continue

            if msg.get("type") == "message":
                content = msg.get("content", "").strip()
                if not content:
                    continue

                response, metadata = await agents[user_id].process_interaction(content)
                await websocket.send_json({"type": "response", "content": response, "metadata": metadata})

                status_payload = {"type": "status", "metadata": agents[user_id].get_status()}
                for conn in list(ws_connections[user_id]):
                    if conn != websocket:
                        try:
                            await conn.send_json(status_payload)
                        except:
                            ws_connections[user_id].discard(conn)

            elif msg.get("type") == "get_status":
                await websocket.send_json({"type": "status", "metadata": agents[user_id].get_status()})

    except WebSocketDisconnect:
        ws_connections[user_id].discard(websocket)
        if not ws_connections[user_id]:
            agents[user_id]._save_state()
            logger.info(f"💾 Saved state for '{user_id}'")
    except Exception as exc:
        logger.error(f"WebSocket error: {exc}")
        ws_connections[user_id].discard(websocket)


# ════════════════════════════════════════════════════════════════
# 🚀 MAIN
# ════════════════════════════════════════════════════════════════

# Убираем отсюда обработчик событий - он должен быть внутри event loop

async def _serve():
    cfg = uvicorn.Config(
        app,
        host=CONFIG.host,
        port=CONFIG.port,
        log_level="info",
    )
    await uvicorn.Server(cfg).serve()

if __name__ == "__main__":
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        print("\n👋 До встречи!")
    except Exception as exc:
        print(f"❌ Ошибка: {exc}")
        traceback.print_exc()