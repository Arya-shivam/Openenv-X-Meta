# Copyright (c) 2026. BankSupportEnv - OpenEnv  Meta Hackathon
# Multi-turn Banking Customer Support RL Environment

"""
Data models for BankSupportEnv.

Defines the typed Pydantic contracts (Action, Observation, State)
that make the environment OpenEnv-compliant.
"""

from typing import Any, Dict, List, Optional
from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class BankSupportAction(Action):
    """
    Action space for the banking support agent.

    The agent can only communicate via free-text responses.
    It cannot take special actions (e.g., 'block card') directly -
    it must express everything through language, which is the
    realistic constraint for a customer support agent.

    Attributes:
        agent_response: Full text response the agent sends to the customer.
                        Must be non-empty, max 1000 characters.
    """

    agent_response: str = ""

    def __post_init__(self):
        if not self.agent_response or not self.agent_response.strip():
            raise ValueError("agent_response must be non-empty")
        if len(self.agent_response) > 1000:
            raise ValueError("agent_response must be at most 1000 characters")


class BankSupportObservation(Observation):
    """
    Observation space for the banking support environment.

    Contains everything the agent can see at each turn.
    Note: account_context is deliberately limited - the agent sees
    account type and join date but NOT full account/card numbers.
    This tests whether the agent handles partial information correctly.

    Attributes:
        task_id: One of 'transaction_dispute', 'card_block', 'loan_enquiry'
        turn: Current turn number (starts at 1)
        customer_message: What the customer just said
        conversation_history: Full chat history as [{role, content}]
        account_context: Account data the agent is ALLOWED to see
        compliance_flags: Rules broken so far this episode
    """

    task_id: str = ""
    turn: int = 1
    customer_message: str = ""
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    account_context: Dict[str, Any] = Field(default_factory=dict)
    compliance_flags: List[str] = Field(default_factory=list)


class BankSupportState(State):
    """
    Internal environment state - hidden from agent, visible to graders.

    Holds the ground truth that the grader needs but the agent cannot see.
    For example, the correct transaction amount for Task 1, or all three
    required clarifying questions for Task 3.

    Attributes:
        task_id: Current task identifier
        scenario: Full scenario including hidden ground truth
        identity_verified: Whether agent has verified customer identity
        issue_identified: Whether agent has identified the core issue
        required_info_collected: List of required info items collected so far
        compliance_violations: List of compliance violations triggered
    """

    task_id: str = ""
    episode_id: str = ""
    step_count: int = 0
    scenario: Dict[str, Any] = Field(default_factory=dict)
    identity_verified: bool = False
    issue_identified: bool = False
    required_info_collected: List[str] = Field(default_factory=list)
    compliance_violations: List[str] = Field(default_factory=list)


if __name__ == "__main__":
    # Quick smoke test - should import and instantiate without error
    action = BankSupportAction(agent_response="Hello, how can I help you?")
    print(f"[PASS] Action: {action.agent_response[:40]}...")

    obs = BankSupportObservation(
        task_id="transaction_dispute",
        turn=1,
        customer_message="I see a charge I don't recognise.",
        done=False,
        reward=0.0,
    )
    print(f"[PASS] Observation: task={obs.task_id}, turn={obs.turn}")

    state = BankSupportState(
        episode_id="test-001",
        step_count=0,
        task_id="transaction_dispute",
        scenario={"opening_message": "Hi"},
    )
    print(f"[PASS] State: task={state.task_id}, verified={state.identity_verified}")
    print("\nAll models OK.")
