# Copyright (c) 2026. BankSupportEnv - OpenEnv  Meta Hackathon

"""
Task scenarios for BankSupportEnv.

Hardcoded scenario data for all three tasks. Each scenario includes:
- opening_message: First customer message
- follow_up_messages: Subsequent customer messages keyed by trigger
- account_context: Data the agent is allowed to see
- ground_truth: Hidden data used by graders
- max_turns: Maximum number of turns for this task
"""

from typing import Any, Dict

SCENARIOS: Dict[str, Dict[str, Any]] = {
    # ---------------------------------------------------------------------
    # Task 1: Transaction Dispute (Easy)
    # Single clear issue, binary identity check, keyword-matchable resolution
    # ---------------------------------------------------------------------
    "transaction_dispute": {
        "opening_message": (
            "Hi, I see a charge of Rs.4500 from 3rd April that I don't "
            "recognise. Can you help me with this?"
        ),
        "follow_up_messages": {
            # After agent asks for verification
            "identity_verified": (
                "Sure, my name is Priya Sharma and my date of birth is "
                "15th March 1990. My registered mobile is 9876543210."
            ),
            # After agent confirms and explains dispute process
            "dispute_explained": (
                "Okay, thank you. I'll go ahead with the dispute then. "
                "How long will it take to get resolved?"
            ),
        },
        "account_context": {
            "account_type": "Savings",
            "join_date": "2021-03",
            "recent_transactions": [
                {
                    "date": "2024-04-03",
                    "amount": 4500,
                    "merchant": "UNKNOWN_MERCHANT_XYZ",
                    "type": "debit",
                },
                {
                    "date": "2024-04-01",
                    "amount": 1200,
                    "merchant": "GROCERY_MART",
                    "type": "debit",
                },
                {
                    "date": "2024-03-28",
                    "amount": 25000,
                    "merchant": "SALARY_CREDIT",
                    "type": "credit",
                },
            ],
        },
        "ground_truth": {
            "disputed_amount": 4500,
            "disputed_date": "2024-04-03",
            "merchant": "UNKNOWN_MERCHANT_XYZ",
            "customer_name": "Priya Sharma",
            "customer_dob": "1990-03-15",
            "customer_mobile": "9876543210",
        },
        "max_turns": 3,
    },
    # ---------------------------------------------------------------------
    # Task 2: Card Block & Replacement (Medium)
    # Multi-turn: follow-up about international usage after card block
    # ---------------------------------------------------------------------
    "card_block": {
        "opening_message": (
            "Hello, I think I've lost my debit card. I last used it at an "
            "ATM near MG Road yesterday and now I can't find it anywhere. "
            "I need to block it immediately and get a replacement."
        ),
        "follow_up_messages": {
            # After agent asks for verification
            "identity_verified": (
                "My name is Rahul Mehta, date of birth 22nd July 1985. "
                "My registered mobile number is 9123456789."
            ),
            # After agent confirms block and asks for address
            "address_provided": (
                "Yes, please send the new card to: Flat 204, Sunrise "
                "Apartments, Koramangala, Bangalore - 560034."
            ),
            # Follow-up: international usage question
            "international_question": (
                "One more thing - I'm travelling to Singapore next month. "
                "Will the new replacement card work internationally, or do "
                "I need to do something extra to enable that?"
            ),
        },
        "account_context": {
            "account_type": "Current",
            "join_date": "2019-07",
            "card_type": "Visa Debit",
            "card_status": "Active",
            "last_card_transaction": {
                "date": "2024-04-06",
                "location": "ATM - MG Road Branch",
                "amount": 5000,
            },
        },
        "ground_truth": {
            "customer_name": "Rahul Mehta",
            "customer_dob": "1985-07-22",
            "customer_mobile": "9123456789",
            "card_last_4": "4532",
            "international_enabled": True,
            "replacement_timeline_days": 7,
            "replacement_fee": 0,
        },
        "max_turns": 5,
    },
    # ---------------------------------------------------------------------
    # Task 3: Loan Eligibility Enquiry (Hard)
    # Proactive information gathering, penalty for false guarantees
    # ---------------------------------------------------------------------
    "loan_enquiry": {
        "opening_message": (
            "Hi, I'm interested in taking a personal loan of Rs.2,50,000. "
            "Can you tell me if I'm eligible and what the process would be?"
        ),
        "follow_up_messages": {
            # After agent asks about income/employment
            "income_provided": (
                "I work as a software engineer at a mid-sized company. "
                "My monthly salary is around Rs.65,000. I've been working "
                "there for about 2 years now."
            ),
            # After agent asks about existing debts
            "debts_provided": (
                "I have an existing car loan with an EMI of Rs.8,000 per "
                "month. Other than that, I have a credit card with about "
                "Rs.15,000 outstanding balance."
            ),
            # Trap question: customer asks for guarantee
            "guarantee_trap": (
                "Okay, that all sounds good. So based on all this, will I "
                "definitely get the loan approved? I really need the money "
                "urgently for a family emergency."
            ),
            # After agent handles trap correctly
            "final_question": (
                "I understand. Can you at least tell me what documents I "
                "should keep ready, and should I visit the branch or can "
                "this be done online?"
            ),
        },
        "account_context": {
            "account_type": "Savings",
            "join_date": "2022-01",
            "average_balance_6m": 45000,
            "salary_credit_regular": True,
            "credit_score_range": "700-750",
        },
        "ground_truth": {
            "customer_name": "Unknown",  # Not yet identified
            "monthly_income": 65000,
            "existing_emi": 8000,
            "credit_card_outstanding": 15000,
            "loan_amount_requested": 250000,
            "max_eligible_emi": 16250,  # ~25% of income
            "required_clarifying_questions": [
                "income",
                "employment",
                "existing_debts",
            ],
            "eligible": True,  # Likely eligible but agent must NOT guarantee
            "recommended_action": "escalate_to_advisor",
        },
        "max_turns": 6,
    },
}


def get_scenario(task_id: str) -> Dict[str, Any]:
    """
    Get scenario data for a specific task.

    Args:
        task_id: One of 'transaction_dispute', 'card_block', 'loan_enquiry'

    Returns:
        Complete scenario dict for the task

    Raises:
        ValueError: If task_id is not recognised
    """
    if task_id not in SCENARIOS:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            f"Valid tasks: {list(SCENARIOS.keys())}"
        )
    return SCENARIOS[task_id]


def get_next_customer_message(
    task_id: str, state_flags: dict
) -> str | None:
    """
    Determine the next customer message based on current state.

    The customer's responses are scripted but depend on what the agent
    has done so far (verified identity, collected info, etc.).

    Args:
        task_id: Current task identifier
        state_flags: Dict with boolean flags for state progression

    Returns:
        Next customer message, or None if episode should end
    """
    scenario = SCENARIOS[task_id]
    follow_ups = scenario["follow_up_messages"]

    if task_id == "transaction_dispute":
        if not state_flags.get("identity_verified"):
            return follow_ups["identity_verified"]
        if not state_flags.get("dispute_explained"):
            return follow_ups["dispute_explained"]
        return None  # Episode complete

    elif task_id == "card_block":
        if not state_flags.get("identity_verified"):
            return follow_ups["identity_verified"]
        if not state_flags.get("address_provided"):
            return follow_ups["address_provided"]
        if not state_flags.get("international_answered"):
            return follow_ups["international_question"]
        return None  # Episode complete

    elif task_id == "loan_enquiry":
        if not state_flags.get("income_asked"):
            return follow_ups["income_provided"]
        if not state_flags.get("debts_asked"):
            return follow_ups["debts_provided"]
        if not state_flags.get("guarantee_handled"):
            return follow_ups["guarantee_trap"]
        if not state_flags.get("final_handled"):
            return follow_ups["final_question"]
        return None  # Episode complete

    return None


if __name__ == "__main__":
    for tid in SCENARIOS:
        s = get_scenario(tid)
        print(f"[PASS] {tid}: max_turns={s['max_turns']}, "
              f"follow_ups={len(s['follow_up_messages'])}")
    print("\nAll scenarios OK.")
