# Copyright (c) 2026. BankSupportEnv - OpenEnv  Meta Hackathon

"""
FastAPI application for BankSupportEnv.

Creates an HTTP server that exposes the BankSupportEnvironment
over HTTP and WebSocket endpoints.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import json
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import environment
try:
    from .environment import BankSupportEnvironment
except ImportError:
    from server.environment import BankSupportEnvironment

import sys
import os

import uvicorn
import gradio as gr
from openenv.core.env_server.web_interface import build_gradio_app, WebInterfaceManager, get_quick_start_markdown, _extract_action_fields, _is_chat_env, load_environment_metadata
from openenv.core.env_server.gradio_theme import OPENENV_GRADIO_CSS, OPENENV_GRADIO_THEME

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models import BankSupportAction, BankSupportObservation

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------

app = FastAPI(
    title="BankSupportEnv",
    description="Multi-turn Banking Customer Support RL Environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage for concurrent environments
_sessions: Dict[str, BankSupportEnvironment] = {}


def _get_or_create_env(session_id: Optional[str] = None) -> tuple:
    """Get existing env session or create a new one."""
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]
    new_id = session_id or str(uuid.uuid4())
    env = BankSupportEnvironment()
    _sessions[new_id] = env
    return new_id, env


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    agent_response: str
    session_id: Optional[str] = None


class ResetResponse(BaseModel):
    session_id: str
    observation: Dict[str, Any]


class StepResponse(BaseModel):
    session_id: str
    observation: Dict[str, Any]
    reward: float
    done: bool


class StateResponse(BaseModel):
    session_id: str
    state: Dict[str, Any]


# -----------------------------------------------------------------------------
# HTTP Endpoints
# -----------------------------------------------------------------------------

@app.get("/")
@app.get("/web")
async def root():
    """Redirect the app root to the Gradio interface."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/web/")



@app.get("/health")
async def health():
    """Health check endpoint. Returns 200 if server is running."""
    return {"status": "healthy", "environment": "bank-support-env"}


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest = ResetRequest()):
    """
    Reset the environment and start a new episode.

    Args:
        request: ResetRequest with optional task_id, seed, session_id

    Returns:
        ResetResponse with session_id and initial observation
    """
    session_id, env = _get_or_create_env(request.session_id)

    obs = env.reset(task_id=request.task_id, seed=request.seed)

    obs_dict = {
        "task_id": obs.task_id,
        "turn": obs.turn,
        "customer_message": obs.customer_message,
        "conversation_history": obs.conversation_history,
        "account_context": obs.account_context,
        "compliance_flags": obs.compliance_flags,
        "done": obs.done,
        "reward": obs.reward,
    }

    return ResetResponse(session_id=session_id, observation=obs_dict)


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """
    Execute one step in the environment.

    Args:
        request: StepRequest with agent_response and session_id

    Returns:
        StepResponse with observation, reward, and done flag
    """
    session_id = request.session_id
    if not session_id or session_id not in _sessions:
        return StepResponse(
            session_id=session_id or "",
            observation={
                "error": "No active session. Call /reset first.",
                "done": True,
                "reward": 0.0,
            },
            reward=0.0,
            done=True,
        )

    env = _sessions[session_id]

    action = {"agent_response": request.agent_response}
    obs = env.step(action)

    obs_dict = {
        "task_id": obs.task_id,
        "turn": obs.turn,
        "customer_message": obs.customer_message,
        "conversation_history": obs.conversation_history,
        "account_context": obs.account_context,
        "compliance_flags": obs.compliance_flags,
        "done": obs.done,
        "reward": obs.reward,
    }

    # Clean up completed sessions
    if obs.done:
        _sessions.pop(session_id, None)

    return StepResponse(
        session_id=session_id,
        observation=obs_dict,
        reward=obs.reward if obs.reward else 0.0,
        done=obs.done if obs.done else False,
    )


@app.get("/state")
async def get_state(session_id: str):
    """
    Get the current internal state for a session.

    Args:
        session_id: Active session identifier

    Returns:
        StateResponse with internal state
    """
    if session_id not in _sessions:
        return {"error": "No active session with that ID."}

    env = _sessions[session_id]
    s = env.state

    state_dict = {
        "episode_id": s.episode_id,
        "step_count": s.step_count,
        "task_id": s.task_id,
        "identity_verified": s.identity_verified,
        "issue_identified": s.issue_identified,
        "required_info_collected": s.required_info_collected,
        "compliance_violations": s.compliance_violations,
    }

    return StateResponse(session_id=session_id, state=state_dict)


@app.get("/docs-info")
async def docs_info():
    """Return environment documentation metadata."""
    return {
        "name": "BankSupportEnv",
        "version": "1.0.0",
        "description": "Multi-turn banking customer support RL environment",
        "tasks": [
            {
                "id": "transaction_dispute",
                "name": "Transaction Dispute Resolution",
                "difficulty": "easy",
                "max_steps": 3,
            },
            {
                "id": "card_block",
                "name": "Card Block and Replacement",
                "difficulty": "medium",
                "max_steps": 5,
            },
            {
                "id": "loan_enquiry",
                "name": "Loan Eligibility Enquiry",
                "difficulty": "hard",
                "max_steps": 6,
            },
        ],
    }


# -----------------------------------------------------------------------------
# WebSocket Endpoint
# -----------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint for real-time environment interaction.

    Protocol:
    - Client sends JSON messages with 'type' field: 'reset' or 'step'
    - Server responds with observation, reward, done
    """
    await ws.accept()
    env = BankSupportEnvironment()
    session_id = str(uuid.uuid4())

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type", "")

            if msg_type == "reset":
                task_id = msg.get("task_id")
                obs = env.reset(task_id=task_id)
                await ws.send_json({
                    "type": "reset_result",
                    "session_id": session_id,
                    "observation": {
                        "task_id": obs.task_id,
                        "turn": obs.turn,
                        "customer_message": obs.customer_message,
                        "conversation_history": obs.conversation_history,
                        "account_context": obs.account_context,
                        "compliance_flags": obs.compliance_flags,
                        "done": obs.done,
                        "reward": obs.reward,
                    },
                })

            elif msg_type == "step":
                agent_response = msg.get("agent_response", "")
                action = {"agent_response": agent_response}
                obs = env.step(action)
                await ws.send_json({
                    "type": "step_result",
                    "observation": {
                        "task_id": obs.task_id,
                        "turn": obs.turn,
                        "customer_message": obs.customer_message,
                        "conversation_history": obs.conversation_history,
                        "account_context": obs.account_context,
                        "compliance_flags": obs.compliance_flags,
                        "done": obs.done,
                        "reward": obs.reward,
                    },
                    "reward": obs.reward,
                    "done": obs.done,
                })

                if obs.done:
                    break

            elif msg_type == "state":
                s = env.state
                await ws.send_json({
                    "type": "state_result",
                    "state": {
                        "episode_id": s.episode_id,
                        "step_count": s.step_count,
                        "task_id": s.task_id,
                        "identity_verified": s.identity_verified,
                        "issue_identified": s.issue_identified,
                    },
                })

            else:
                await ws.send_json({"error": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        await ws.send_json({"error": str(e)})
    finally:
        env.close()

# -----------------------------------------------------------------------------
# Gradio Web Interface Mounting
# -----------------------------------------------------------------------------

# Initialize the environment for the web manager
global_env = BankSupportEnvironment()
metadata = load_environment_metadata(global_env, "bank-support-env")
web_manager = WebInterfaceManager(global_env, BankSupportAction, BankSupportObservation, metadata)

action_fields = _extract_action_fields(BankSupportAction)
is_chat_env = _is_chat_env(BankSupportAction)
quick_start_md = get_quick_start_markdown(metadata, BankSupportAction, BankSupportObservation)

gradio_blocks = build_gradio_app(
    web_manager,
    action_fields,
    metadata,
    is_chat_env,
    title="BankSupportEnv",
    quick_start_md=quick_start_md,
)

app = gr.mount_gradio_app(
    app,
    gradio_blocks,
    path="/web",
    theme=OPENENV_GRADIO_THEME,
    css=OPENENV_GRADIO_CSS,
)


import logging
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Direct execution
# -----------------------------------------------------------------------------


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
