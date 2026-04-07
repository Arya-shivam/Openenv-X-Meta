# Copyright (c) 2026. BankSupportEnv - OpenEnv  Meta Hackathon

"""
Client for BankSupportEnv.

This is what external users import to interact with the environment.
It communicates with the FastAPI server over HTTP (or WebSocket)
and translates between typed Pydantic models and the wire format.

Usage:
    from client import BankSupportEnv
    from models import BankSupportAction

    # HTTP client
    import requests

    env = BankSupportEnv(base_url="http://localhost:8000")
    obs = env.reset(task_id="transaction_dispute")
    print(obs.customer_message)

    action = BankSupportAction(agent_response="Please verify your identity.")
    obs = env.step(action)
    print(obs.reward, obs.done)
"""

import json
from typing import Any, Dict, Optional

import requests

from models import BankSupportAction, BankSupportObservation, BankSupportState


class StepResult:
    """Result from an environment step."""

    def __init__(
        self,
        observation: BankSupportObservation,
        reward: float,
        done: bool,
    ):
        self.observation = observation
        self.reward = reward
        self.done = done


class BankSupportEnv:
    """
    Client for the BankSupportEnv environment.

    Connects to the FastAPI server and provides a typed Python interface
    for interacting with the environment.

    Args:
        base_url: URL of the running environment server
        session_id: Optional session ID to resume

    Example:
        >>> env = BankSupportEnv(base_url="http://localhost:8000")
        >>> obs = env.reset(task_id="transaction_dispute")
        >>> print(obs.customer_message)
        >>> result = env.step(BankSupportAction(agent_response="Hello!"))
        >>> print(result.reward, result.done)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        session_id: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> BankSupportObservation:
        """
        Reset the environment and start a new episode.

        Args:
            task_id: Which task to run (defaults to 'transaction_dispute')
            seed: Optional random seed

        Returns:
            BankSupportObservation with the first customer message
        """
        payload = {}
        if task_id:
            payload["task_id"] = task_id
        if seed is not None:
            payload["seed"] = seed
        if self.session_id:
            payload["session_id"] = self.session_id

        resp = requests.post(f"{self.base_url}/reset", json=payload)
        resp.raise_for_status()
        data = resp.json()

        self.session_id = data.get("session_id")
        return self._parse_observation(data.get("observation", {}))

    def step(self, action: BankSupportAction) -> StepResult:
        """
        Execute one step in the environment.

        Args:
            action: BankSupportAction with agent_response

        Returns:
            StepResult with observation, reward, done
        """
        payload = self._step_payload(action)

        resp = requests.post(f"{self.base_url}/step", json=payload)
        resp.raise_for_status()
        data = resp.json()

        return self._parse_result(data)

    def get_state(self) -> BankSupportState:
        """
        Get the current internal state.

        Returns:
            BankSupportState
        """
        resp = requests.get(
            f"{self.base_url}/state",
            params={"session_id": self.session_id},
        )
        resp.raise_for_status()
        data = resp.json()
        return self._parse_state(data.get("state", {}))

    def close(self):
        """Clean up the client session."""
        self.session_id = None

    # -- Context manager support --------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # -- Wire format translation (Module 4 pattern) ------------------

    def _step_payload(self, action: BankSupportAction) -> dict:
        """Convert typed action to wire format."""
        return {
            "agent_response": action.agent_response,
            "session_id": self.session_id,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        """Parse wire format into typed StepResult."""
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=self._parse_observation(obs_data),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_observation(self, obs_data: dict) -> BankSupportObservation:
        """Parse observation dict into typed BankSupportObservation."""
        return BankSupportObservation(
            task_id=obs_data.get("task_id", ""),
            turn=obs_data.get("turn", 1),
            customer_message=obs_data.get("customer_message", ""),
            conversation_history=obs_data.get("conversation_history", []),
            account_context=obs_data.get("account_context", {}),
            compliance_flags=obs_data.get("compliance_flags", []),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward", 0.0),
        )

    def _parse_state(self, payload: dict) -> BankSupportState:
        """Parse state dict into typed BankSupportState."""
        return BankSupportState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            scenario=payload.get("scenario", {}),
            identity_verified=payload.get("identity_verified", False),
            issue_identified=payload.get("issue_identified", False),
            required_info_collected=payload.get("required_info_collected", []),
            compliance_violations=payload.get("compliance_violations", []),
        )


# -----------------------------------------------------------------------------
# Docker / HF Space helpers
# -----------------------------------------------------------------------------


def from_docker_image(image_name: str, port: int = 8000) -> BankSupportEnv:
    """
    Create a client connected to a locally running Docker container.

    Args:
        image_name: Docker image name
        port: Port to expose

    Returns:
        BankSupportEnv client
    """
    return BankSupportEnv(base_url=f"http://localhost:{port}")


def from_hf_space(space_url: str) -> BankSupportEnv:
    """
    Create a client connected to a Hugging Face Space.

    Args:
        space_url: Full URL of the HF Space

    Returns:
        BankSupportEnv client
    """
    return BankSupportEnv(base_url=space_url)
