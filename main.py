"""
🌐 main.py — FastAPI server: routes, WebSocket, lifespan
"""

import json
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional, Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from config import CONFIG
from logging_setup import setup_logging
from models.teacher import TeacherLLM
from agent import AdvancedAutonomousAgent

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logger = setup_logging(CONFIG.base_dir)

# ── Global state ───────────────────────────────────────────────────────────────
teacher: Optional[TeacherLLM] = None
agents: Dict[str, AdvancedAutonomousAgent] = {}
ws_connections: Dict[str, Set[WebSocket]] = {}

STATIC_DIR = Path(__file__).parent / "static"


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global teacher
    teacher = TeacherLLM(CONFIG.lm_studio_url, CONFIG.lm_studio_key)
    await teacher.connect()
    logger.info(f"🚀 Server started — {CONFIG.host}:{CONFIG.port} [{CONFIG.device.upper()}]")
    _print_banner()
    yield
    # Shutdown
    for agent in agents.values():
        agent._save_state()
    if teacher:
        await teacher.close()
    logger.info("👋 Server shutdown — all states saved")


def _print_banner() -> None:
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


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Advanced AI Agent v4",
    description="Автономный обучающийся агент с Knowledge Distillation",
    version="4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML UI)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── HTTP routes ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": CONFIG.device,
        "model": f"{CONFIG.n_layers}L-{CONFIG.d_model}D-{CONFIG.n_heads}H",
        "active_agents": len(agents),
    }


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


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    user_id = websocket.query_params.get("user_id", f"user_{int(time.time())}")

    # Create agent on first connection
    if user_id not in agents:
        agents[user_id] = AdvancedAutonomousAgent(user_id, teacher)

    ws_connections.setdefault(user_id, set()).add(websocket)

    try:
        # Send initial status
        await websocket.send_json({"type": "status", "metadata": agents[user_id].get_status()})

        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "content": "Invalid JSON"})
                continue

            msg_type = msg.get("type")

            if msg_type == "message":
                content = msg.get("content", "").strip()
                if not content:
                    continue

                response, metadata = await agents[user_id].process_interaction(content)
                await websocket.send_json({
                    "type": "response",
                    "content": response,
                    "metadata": metadata,
                })

                # Broadcast updated status to other tabs
                status_payload = {"type": "status", "metadata": agents[user_id].get_status()}
                for conn in list(ws_connections[user_id]):
                    if conn is not websocket:
                        try:
                            await conn.send_json(status_payload)
                        except Exception:
                            ws_connections[user_id].discard(conn)

            elif msg_type == "get_status":
                await websocket.send_json({
                    "type": "status",
                    "metadata": agents[user_id].get_status(),
                })

    except WebSocketDisconnect:
        ws_connections[user_id].discard(websocket)
        if not ws_connections[user_id]:
            agents[user_id]._save_state()
            logger.info(f"💾 Auto-saved state for '{user_id}' on disconnect")
    except Exception as exc:
        logger.error(f"WebSocket error [{user_id}]: {exc}")
        ws_connections[user_id].discard(websocket)


# ── Entry point ────────────────────────────────────────────────────────────────

async def _serve() -> None:
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
        print(f"❌ Ошибка запуска: {exc}")
        import traceback
        traceback.print_exc()
