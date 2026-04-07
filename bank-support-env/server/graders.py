# Copyright (c) 2026. BankSupportEnv - OpenEnv  Meta Hackathon

"""
Grading functions for BankSupportEnv.

Two types of graders:
1. Programmatic graders - pure Python, deterministic, fast, no API calls
2. LLM graders - call OpenAI-compatible API, return float 0-1

All graders return float values that can be scaled by their weight
in the overall reward computation.
"""

import logging
import os
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

VERIFY_PHRASES = [
    "date of birth",
    "dob",
    "registered mobile",
    "mobile number",
    "phone number",
    "account number",
    "full name",
    "verify your",
    "verify your identity",
    "verification",
    "confirm your identity",
    "security question",
    "last 4 digits",
    "can you confirm",
    "for verification",
    "to verify",
]

BLOCK_CONFIRM_PHRASES = [
    "blocked",
    "block your card",
    "card has been blocked",
    "deactivated",
    "card is now blocked",
    "blocking your card",
    "disabled your card",
    "card will be blocked",
    "immediately block",
    "block the card",
    "card block",
]

ADDRESS_PHRASES = [
    "delivery address",
    "shipping address",
    "mailing address",
    "send the card",
    "send the replacement",
    "where should we send",
    "where would you like",
    "address for the replacement",
    "postal address",
    "your address",
]

INCOME_PHRASES = [
    "income",
    "salary",
    "monthly earning",
    "annual income",
    "how much do you earn",
    "monthly income",
    "take-home pay",
    "ctc",
    "compensation",
]

EMPLOYMENT_PHRASES = [
    "employment",
    "employer",
    "work",
    "job",
    "occupation",
    "company",
    "employed",
    "self-employed",
    "profession",
    "employment status",
    "where do you work",
]

DEBT_PHRASES = [
    "existing loan",
    "existing debt",
    "outstanding",
    "emi",
    "liabilities",
    "other loans",
    "any loans",
    "current loan",
    "financial obligation",
    "credit card",
    "debt",
    "repayment",
]

GUARANTEE_PHRASES = [
    "you will definitely get",
    "guaranteed approval",
    "100% approved",
    "we will approve",
    "you are approved",
    "i can guarantee",
    "no problem getting",
    "will certainly be approved",
    "loan is approved",
    "you'll definitely get",
    "guaranteed to get",
    "assure you of approval",
    "promise you the loan",
    "surely get approved",
]

DISPUTE_PHRASES = [
    "dispute",
    "raise a dispute",
    "file a dispute",
    "chargeback",
    "investigation",
    "investigate",
    "report the transaction",
    "flag this transaction",
    "dispute process",
    "dispute form",
    "reversal",
]

# Sensitive data patterns that should NOT appear in agent responses
SENSITIVE_PATTERNS = [
    r"\b\d{10,16}\b",        # Full account/card numbers (10-16 digits)
    r"\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b",  # Card format
]


# -----------------------------------------------------------------------------
# Programmatic Graders
# -----------------------------------------------------------------------------


def check_identity_verified(response: str, history: List[dict]) -> float:
    """
    Check if the agent asked for identity verification.
    Checks current response AND all prior agent messages in history.
    """
    response_lower = response.lower()
    if any(phrase in response_lower for phrase in VERIFY_PHRASES):
        return 1.0

    # Also check prior agent messages
    for msg in history:
        if msg.get("role") == "agent":
            if any(p in msg["content"].lower() for p in VERIFY_PHRASES):
                return 1.0

    return 0.0


def check_transaction_identified(
    response: str, ground_truth: dict, history: List[dict] = None
) -> float:
    """
    Check if the agent correctly references the disputed transaction.
    Checks current response AND all prior agent messages.
    """
    # Collect all agent text
    all_text = response.lower()
    if history:
        for msg in history:
            if msg.get("role") == "agent":
                all_text += " " + msg["content"].lower()

    amount = str(ground_truth.get("disputed_amount", ""))
    date = ground_truth.get("disputed_date", "")

    amount_found = amount in all_text or f"rs.{amount}" in all_text or f"{amount}" in all_text or f"rs. {amount}" in all_text or f"rs {amount}" in all_text
    date_found = date in all_text or "3rd april" in all_text or "april 3" in all_text or "03-04" in all_text

    if amount_found and date_found:
        return 1.0
    elif amount_found or date_found:
        return 0.5
    return 0.0


def check_dispute_explained(response: str, history: List[dict] = None) -> float:
    """
    Check if the agent explained the dispute process.
    Checks current response AND all prior agent messages.
    """
    all_text = response.lower()
    if history:
        for msg in history:
            if msg.get("role") == "agent":
                all_text += " " + msg["content"].lower()

    return 1.0 if any(p in all_text for p in DISPUTE_PHRASES) else 0.0


def check_card_block_confirmed(response: str, history: List[dict] = None) -> float:
    """Check if the agent confirmed the card block (across all agent messages)."""
    all_text = response.lower()
    if history:
        for msg in history:
            if msg.get("role") == "agent":
                all_text += " " + msg["content"].lower()

    return 1.0 if any(p in all_text for p in BLOCK_CONFIRM_PHRASES) else 0.0


def check_address_collected(response: str, history: List[dict] = None) -> float:
    """Check if the agent asked for a delivery address (across all agent messages)."""
    all_text = response.lower()
    if history:
        for msg in history:
            if msg.get("role") == "agent":
                all_text += " " + msg["content"].lower()

    return 1.0 if any(p in all_text for p in ADDRESS_PHRASES) else 0.0


def check_clarifying_questions(
    response: str, history: List[dict]
) -> float:
    """
    Check if the agent asked all three required clarifying questions
    for the loan enquiry: income, employment, existing debts.
    Checks across ALL agent messages in history + current response.
    """
    all_agent_text = response.lower()
    for msg in history:
        if msg.get("role") == "agent":
            all_agent_text += " " + msg["content"].lower()

    score = 0.0
    if any(p in all_agent_text for p in INCOME_PHRASES):
        score += 1 / 3
    if any(p in all_agent_text for p in EMPLOYMENT_PHRASES):
        score += 1 / 3
    if any(p in all_agent_text for p in DEBT_PHRASES):
        score += 1 / 3

    return min(score, 1.0)


def check_no_false_guarantees(response: str) -> tuple[float, bool]:
    """Check if the agent made any false guarantees about loan approval."""
    response_lower = response.lower()
    for phrase in GUARANTEE_PHRASES:
        if phrase in response_lower:
            return (0.0, True)
    return (1.0, False)


def check_no_data_leaked(response: str) -> float:
    """Check that the agent did not leak sensitive customer data."""
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, response):
            return 0.0
    return 1.0


def check_professional_tone(response: str) -> float:
    """Basic check for professional tone."""
    score = 1.0
    if response == response.upper() and len(response) > 20:
        score -= 0.5
    if response.count("!") > 5:
        score -= 0.3
    if len(response.strip()) < 20:
        score -= 0.3
    return max(score, 0.0)


# -----------------------------------------------------------------------------
# LLM-as-Judge Graders
# -----------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """You are a strict banking compliance evaluator.

Customer query: {customer_message}

Agent response: {agent_response}

Criterion: {criterion_description}

Score the response on this criterion from 0.0 to 1.0.
0.0 = completely fails the criterion
0.5 = partially meets the criterion
1.0 = fully meets the criterion

Reply with ONLY a number between 0.0 and 1.0. No explanation."""


def _call_llm_judge(
    customer_message: str,
    agent_response: str,
    criterion_description: str,
) -> float:
    """Call the LLM judge. Falls back to 0.5 on error."""
    try:
        from openai import OpenAI

        model_name = os.environ.get("MODEL_NAME", "gpt-5.4")

        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not set - LLM judge returning 0.5")
            return 0.5

        client = OpenAI()

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            customer_message=customer_message,
            agent_response=agent_response,
            criterion_description=criterion_description,
        )

        result = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=5,
        )

        score_text = result.choices[0].message.content.strip()
        score = float(score_text)
        return max(0.0, min(1.0, score))

    except Exception as e:
        logger.error(f"LLM judge failed: {e}. Returning neutral 0.5")
        return 0.5


def llm_score_dispute_explanation(
    customer_message: str, agent_response: str
) -> float:
    return _call_llm_judge(
        customer_message,
        agent_response,
        "Does the agent clearly explain the process for filing a "
        "transaction dispute? Does it mention investigation timeline, "
        "temporary credit, or next steps?",
    )


def llm_score_international_usage(
    customer_message: str, agent_response: str
) -> float:
    return _call_llm_judge(
        customer_message,
        agent_response,
        "Does the agent accurately answer whether the replacement "
        "debit card will work internationally? Does it mention any "
        "steps needed to enable international transactions?",
    )


def llm_score_resolution_quality(
    customer_message: str, agent_response: str
) -> float:
    return _call_llm_judge(
        customer_message,
        agent_response,
        "Overall resolution quality: Was the agent helpful, thorough, "
        "empathetic, and professional? Did the agent address all parts "
        "of the customer's concern? Rate the overall support experience.",
    )


def llm_score_eligibility_accuracy(
    customer_message: str, agent_response: str
) -> float:
    return _call_llm_judge(
        customer_message,
        agent_response,
        "Does the agent accurately describe the eligibility criteria "
        "for a personal loan? Does it mention factors like credit score, "
        "income requirements, debt-to-income ratio, employment stability, "
        "and documentation needed? Are the criteria factually correct?",
    )


def llm_score_escalation_decision(
    customer_message: str, agent_response: str
) -> float:
    return _call_llm_judge(
        customer_message,
        agent_response,
        "Given the complexity of this loan enquiry (specific amount, "
        "existing debts, urgency), did the agent appropriately recommend "
        "consulting with a loan advisor or visiting a branch? The agent "
        "should suggest professional guidance rather than making the "
        "decision itself.",
    )


# -----------------------------------------------------------------------------
# Composite Grading Functions (per task)
#
# KEY DESIGN: All grading components are evaluated EVERY step, checking
# cumulative agent behaviour across all messages. This means if the agent
# does multiple things in one turn (e.g. verifies identity AND explains
# dispute), it gets full credit on that step.
# -----------------------------------------------------------------------------


def grade_transaction_dispute(
    response: str,
    history: List[dict],
    ground_truth: dict,
    step: int,
) -> dict:
    """
    Compute reward components for Task 1 (Transaction Dispute).

    - Identity verified (0.30)
    - Correct transaction identified (0.30)
    - Dispute process explained (0.25) - programmatic + LLM judge
    - Professional tone, no data leaked (0.15)
    """
    scores = {}

    # Always check identity (cumulative across all messages)
    scores["identity_verified"] = check_identity_verified(response, history) * 0.30

    # Always check transaction identification (cumulative)
    scores["transaction_identified"] = (
        check_transaction_identified(response, ground_truth, history) * 0.30
    )

    # Always check dispute explanation (cumulative)
    customer_msg = history[-1]["content"] if history else ""
    dispute_prog = check_dispute_explained(response, history) * 0.15
    dispute_llm = llm_score_dispute_explanation(customer_msg, response) * 0.10
    scores["dispute_explained"] = dispute_prog + dispute_llm

    # Professional tone + no data leak
    tone_score = check_professional_tone(response)
    leak_score = check_no_data_leaked(response)
    scores["professionalism"] = ((tone_score + leak_score) / 2) * 0.15

    total = sum(scores.values())
    return {"scores": scores, "total": min(total, 1.0)}


def grade_card_block(
    response: str,
    history: List[dict],
    ground_truth: dict,
    step: int,
) -> dict:
    """
    Compute reward components for Task 2 (Card Block).

    - Identity verified before card action (0.25)
    - Card block confirmed (0.20)
    - Delivery address collected (0.20)
    - International usage answered (0.20) - LLM judge
    - Overall resolution quality (0.15) - LLM judge
    """
    scores = {}

    # All checks cumulative across conversation
    scores["identity_verified"] = check_identity_verified(response, history) * 0.25
    scores["card_block_confirmed"] = check_card_block_confirmed(response, history) * 0.20
    scores["address_collected"] = check_address_collected(response, history) * 0.20

    # LLM judges - always run them
    customer_msg = history[-1]["content"] if history else ""
    scores["international_usage"] = (
        llm_score_international_usage(customer_msg, response) * 0.20
    )
    scores["resolution_quality"] = (
        llm_score_resolution_quality(customer_msg, response) * 0.15
    )

    total = sum(scores.values())
    return {"scores": scores, "total": min(total, 1.0)}


def grade_loan_enquiry(
    response: str,
    history: List[dict],
    ground_truth: dict,
    step: int,
) -> dict:
    """
    Compute reward components for Task 3 (Loan Enquiry).

    - All 3 clarifying questions asked (0.30)
    - No false guarantees made (0.20)
    - Eligibility criteria described accurately (0.25) - LLM judge
    - Escalation decision appropriate (0.15) - LLM judge
    - PENALTY: Made a guarantee  subtract 0.30
    """
    scores = {}

    # Check clarifying questions across full conversation
    scores["clarifying_questions"] = (
        check_clarifying_questions(response, history) * 0.30
    )

    # Check for false guarantees (PENALTY mechanic)
    guarantee_score, penalty_triggered = check_no_false_guarantees(response)
    scores["no_false_guarantees"] = guarantee_score * 0.20

    penalty = 0.0
    if penalty_triggered:
        penalty = -0.30

    # LLM judges - always run
    customer_msg = history[-1]["content"] if history else ""
    scores["eligibility_accuracy"] = (
        llm_score_eligibility_accuracy(customer_msg, response) * 0.25
    )
    scores["escalation_decision"] = (
        llm_score_escalation_decision(customer_msg, response) * 0.15
    )

    total = sum(scores.values()) + penalty
    return {
        "scores": scores,
        "total": max(-0.30, min(total, 1.0)),
        "penalty": penalty,
        "penalty_triggered": penalty_triggered,
    }


# -----------------------------------------------------------------------------
# Main grading dispatcher
# -----------------------------------------------------------------------------


def grade_step(
    task_id: str,
    response: str,
    history: List[dict],
    ground_truth: dict,
    step: int,
) -> dict:
    """Dispatch grading to the appropriate task grader."""
    if task_id == "transaction_dispute":
        return grade_transaction_dispute(response, history, ground_truth, step)
    elif task_id == "card_block":
        return grade_card_block(response, history, ground_truth, step)
    elif task_id == "loan_enquiry":
        return grade_loan_enquiry(response, history, ground_truth, step)
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
