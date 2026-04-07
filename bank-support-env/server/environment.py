# Copyright (c) 2026. BankSupportEnv - OpenEnv  Meta Hackathon

"""
Core environment logic for BankSupportEnv.

Implements reset(), step(), and state property for OpenEnv compliance.
Manages multi-turn conversations with scripted customer follow-ups.
"""

import logging
from typing import Any, Optional
from uuid import uuid4

# Use relative imports when running as package, absolute for standalone
try:
    from .graders import (
        check_address_collected,
        check_card_block_confirmed,
        check_clarifying_questions,
        check_identity_verified,
        grade_step,
        DISPUTE_PHRASES, INCOME_PHRASES, EMPLOYMENT_PHRASES, DEBT_PHRASES,
    )
    from .tasks import SCENARIOS, get_next_customer_message, get_scenario
except ImportError:
    from server.graders import (
        check_address_collected,
        check_card_block_confirmed,
        check_clarifying_questions,
        check_identity_verified,
        grade_step,
        DISPUTE_PHRASES, INCOME_PHRASES, EMPLOYMENT_PHRASES, DEBT_PHRASES,
    )
    from server.tasks import SCENARIOS, get_next_customer_message, get_scenario

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models import BankSupportAction, BankSupportObservation, BankSupportState

logger = logging.getLogger(__name__)


from openenv.core.env_server.interfaces import Environment

class BankSupportEnvironment(Environment):
    """
    Multi-turn Banking Customer Support RL Environment.

    Simulates a banking customer support desk where an AI agent must
    resolve real customer issues correctly, safely, and professionally.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state: Optional[BankSupportState] = None
        self._conversation_history: list = []
        self._step_rewards: list = []
        self._state_flags: dict = {}
        self._follow_up_queue: list = []

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> BankSupportObservation:
        """Start a new episode."""
        if task_id is None:
            task_id = "transaction_dispute"

        scenario = get_scenario(task_id)

        self._state = BankSupportState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=task_id,
            scenario=scenario,
            identity_verified=False,
            issue_identified=False,
            required_info_collected=[],
            compliance_violations=[],
        )

        self._conversation_history = [
            {"role": "customer", "content": scenario["opening_message"]},
        ]

        self._state_flags = {
            "identity_verified": False,
            "dispute_explained": False,
            "card_block_confirmed": False,
            "address_provided": False,
            "international_answered": False,
            "income_asked": False,
            "debts_asked": False,
            "guarantee_handled": False,
            "final_handled": False,
        }

        # Pre-build the follow-up queue based on task
        # This ensures the conversation ALWAYS runs all scripted turns
        follow_ups = scenario.get("follow_up_messages", {})
        if task_id == "transaction_dispute":
            self._follow_up_queue = [
                follow_ups.get("identity_verified", ""),
                follow_ups.get("dispute_explained", ""),
            ]
        elif task_id == "card_block":
            self._follow_up_queue = [
                follow_ups.get("identity_verified", ""),
                follow_ups.get("address_provided", ""),
                follow_ups.get("international_question", ""),
            ]
        elif task_id == "loan_enquiry":
            self._follow_up_queue = [
                follow_ups.get("income_provided", ""),
                follow_ups.get("debts_provided", ""),
                follow_ups.get("guarantee_trap", ""),
                follow_ups.get("final_question", ""),
            ]

        self._step_rewards = []

        return BankSupportObservation(
            task_id=task_id,
            turn=1,
            customer_message=scenario["opening_message"],
            conversation_history=list(self._conversation_history),
            account_context=scenario["account_context"],
            compliance_flags=[],
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: Any,
        **kwargs: Any,
    ) -> BankSupportObservation:
        """Process one agent action and return the next observation."""
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        # Handle both typed action and dict payload
        if isinstance(action, dict):
            agent_response = action.get("agent_response", "")
        elif isinstance(action, BankSupportAction):
            agent_response = action.agent_response
        elif hasattr(action, "agent_response"):
            agent_response = action.agent_response
        else:
            agent_response = str(action)

        if not agent_response or not agent_response.strip():
            agent_response = "I'm here to help."

        # Increment step count
        self._state.step_count += 1
        current_step = self._state.step_count
        task_id = self._state.task_id
        scenario = self._state.scenario

        # Add agent response to conversation history
        self._conversation_history.append({
            "role": "agent",
            "content": agent_response,
        })

        # -- Update state flags FIRST (before grading) ----------------
        self._update_state_flags(agent_response, task_id)

        # -- Run graders ----------------------------------------------
        grade_result = grade_step(
            task_id=task_id,
            response=agent_response,
            history=self._conversation_history,
            ground_truth=scenario.get("ground_truth", {}),
            step=current_step,
        )

        step_reward = grade_result["total"]
        self._step_rewards.append(step_reward)

        # Handle penalty (Task 3)
        if grade_result.get("penalty_triggered"):
            if "false_guarantee" not in self._state.compliance_violations:
                self._state.compliance_violations.append("false_guarantee")

        # -- Generate next customer message ---------------------------
        # Use the pre-built queue: pop the next follow-up in order
        next_customer_message = None
        done = False

        if self._follow_up_queue:
            next_customer_message = self._follow_up_queue.pop(0)
        else:
            done = True  # No more scripted messages

        # Also enforce max_turns
        max_turns = scenario.get("max_turns", 6)
        if current_step >= max_turns:
            done = True
            next_customer_message = None

        if not done and next_customer_message:
            self._conversation_history.append({
                "role": "customer",
                "content": next_customer_message,
            })

        # Build observation
        obs = BankSupportObservation(
            task_id=task_id,
            turn=current_step + 1,
            customer_message=next_customer_message or "",
            conversation_history=list(self._conversation_history),
            account_context=scenario.get("account_context", {}),
            compliance_flags=list(self._state.compliance_violations),
            done=done,
            reward=step_reward,
        )

        return obs

    def _update_state_flags(self, response: str, task_id: str):
        """Update conversation state flags based on agent's response."""
        response_lower = response.lower()

        # Identity verification - all tasks
        identity_score = check_identity_verified(response, self._conversation_history)
        if identity_score > 0:
            self._state_flags["identity_verified"] = True
            self._state.identity_verified = True

        if task_id == "transaction_dispute":
            if any(p in response_lower for p in DISPUTE_PHRASES):
                self._state_flags["dispute_explained"] = True
                self._state.issue_identified = True

        elif task_id == "card_block":
            if check_card_block_confirmed(response) > 0:
                self._state_flags["card_block_confirmed"] = True
                self._state.issue_identified = True

            if check_address_collected(response) > 0:
                self._state_flags["address_provided"] = True
                if "delivery_address" not in self._state.required_info_collected:
                    self._state.required_info_collected.append("delivery_address")

            if "international" in response_lower or "abroad" in response_lower or "overseas" in response_lower:
                self._state_flags["international_answered"] = True

        elif task_id == "loan_enquiry":
            if any(p in response_lower for p in INCOME_PHRASES):
                self._state_flags["income_asked"] = True
                if "income" not in self._state.required_info_collected:
                    self._state.required_info_collected.append("income")

            if any(p in response_lower for p in EMPLOYMENT_PHRASES):
                if "employment" not in self._state.required_info_collected:
                    self._state.required_info_collected.append("employment")

            if any(p in response_lower for p in DEBT_PHRASES):
                self._state_flags["debts_asked"] = True
                if "debts" not in self._state.required_info_collected:
                    self._state.required_info_collected.append("debts")

            last_customer = ""
            for msg in reversed(self._conversation_history):
                if msg["role"] == "customer":
                    last_customer = msg["content"].lower()
                    break

            if "definitely" in last_customer or "guarantee" in last_customer:
                self._state_flags["guarantee_handled"] = True

            if "document" in response_lower or "branch" in response_lower or "online" in response_lower:
                self._state_flags["final_handled"] = True

    @property
    def state(self) -> BankSupportState:
        if self._state is None:
            return BankSupportState(
                episode_id="",
                step_count=0,
                task_id="",
                scenario={},
            )
        return self._state

    def close(self):
        self._state = None
        self._conversation_history = []
        self._step_rewards = []
        self._state_flags = {}
        self._follow_up_queue = []

    def get_episode_score(self) -> float:
        if not self._step_rewards:
            return 0.0
        total = sum(self._step_rewards) / len(self._step_rewards)
        if self._state and self._state.task_id == "loan_enquiry":
            return max(-0.30, min(total, 1.0))
        return max(0.0, min(total, 1.0))
