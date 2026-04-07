
**BankSupportEnv**

Product Requirements Document

OpenEnv × Meta Hackathon — Round 1 Submission

*A Multi-Turn Banking Customer Support RL Environment*

|**Quick Reference**|
| :- |
|Environment: BankSupportEnv|
|Domain: Banking / Fintech Customer Support|
|Tasks: 3  (Transaction Dispute → Card Block → Loan Enquiry)|
|Reward Range: 0.0 – 1.0 per episode|
|Scoring: Programmatic checks + LLM-as-judge|
|Deployment: Hugging Face Space + Docker|
|Inference Script: inference.py (OpenAI client, structured stdout)|


# **1. Project Overview**
BankSupportEnv is a multi-turn reinforcement learning environment built on the OpenEnv framework. It simulates a banking customer support desk where an AI agent must resolve real customer issues correctly, safely, and professionally.

The environment is designed to train and evaluate agents on the exact kind of nuanced, high-stakes communication that makes banking support genuinely hard: strict compliance rules (never share data unprompted, always verify identity), partial information problems (customer doesn’t know their account number), and multi-turn reasoning (ask the right clarifying questions before acting).

**Why banking support?**

- Real-world utility is the highest-weighted judging criterion (30%). Banking support is an immediate, high-value deployment target for LLM agents.
- Deterministic graders are easy to write: ‘did agent verify identity before acting?’ is a binary check.
- LLM scoring is naturally justified: ‘was the resolution actually helpful?’ requires language understanding.
- The penalty mechanic (making false guarantees on loans) demonstrates advanced reward shaping that judges look for.
- The domain has never appeared in the existing OpenEnv Hub — satisfying the novelty criterion.

# **2. Judging Criteria Alignment**
Every design decision in this document maps to one of the five judging criteria. This section makes that mapping explicit.

|**Criterion**|**How BankSupportEnv addresses it**|**Target**|
| :- | :- | :- |
|Real-world utility (30%)|Banking support resolves real customer pain. Agents trained here could deploy directly to production.|Score target: 26–30|
|Task & grader quality (25%)|3 tasks with clear easy→medium→hard progression. Graders are deterministic and produce varied scores.|Score target: 20–25|
|Environment design (20%)|Clean Pydantic models. Reward signals partial progress every turn. Sensible episode boundaries.|Score target: 16–20|
|Code quality (15%)|Follows Module 4 pattern exactly. openenv validate passes. Dockerfile builds clean.|Score target: 12–15|
|Creativity & novelty (10%)|Penalty reward for false guarantees is novel. Banking domain not in Hub.|Score target: 8–10|

**Disqualification risks and mitigations**

|**Risk**|**Mitigation**|
| :- | :- |
|**Environment does not deploy**|Dockerfile tested locally before submission. HF Space health endpoint returns 200.|
|**Graders always return same score**|Each grader has 4–5 independent criteria producing varied scores. Tested with diverse agent outputs.|
|**No baseline inference script**|inference.py in root, uses OpenAI client, emits exact [START]/[STEP]/[END] format.|
|**Plagiarised environment**|Banking support domain is original. Not a modification of any existing OpenEnv environment.|

# **3. Project File Structure**
Following the Module 4 three-component pattern exactly: models, server (environment + FastAPI), client, config.

|**bank-support-env/**|
| :- |
|├── models.py                   ← Pydantic: Action, Observation, State|
|├── client.py                   ← EnvClient subclass (what users import)|
|├── inference.py                ← MANDATORY: baseline script, root level|
|├── openenv.yaml                ← Manifest: tasks, metadata, grader refs|
|├── pyproject.toml              ← Package metadata|
|├── README.md                   ← Full documentation (required)|
|└── server/|
|`    `├── environment.py          ← Core logic: reset(), step(), state, graders|
|`    `├── graders.py              ← All reward logic (programmatic + LLM judge)|
|`    `├── tasks.py                ← 3 task scenarios with hardcoded data|
|`    `├── app.py                  ← FastAPI: create\_fastapi\_app(BankSupportEnv)|
|`    `├── requirements.txt|
|`    `└── Dockerfile|

Critical constraint from the spec: inference.py MUST be in the root directory, not inside server/. The validation script checks for it at the root level.

# **4. Data Models (models.py)**
These are the typed Pydantic contracts that make the environment OpenEnv-compliant. Typed models are required by the spec and checked by openenv validate.

**4.1  BankSupportAction**

|**class BankSupportAction(Action):**|
| :- |
|`    `agent\_response: str      # Full text response the agent sends to customer|
|`    `# Validation: must be non-empty, max 1000 chars|

The action space is intentionally simple: free-text. This is appropriate for a language agent. The agent cannot take any special actions (like ‘block card’) — it can only communicate. This forces the agent to learn through language, which is the realistic constraint.

**4.2  BankSupportObservation**

|**class BankSupportObservation(Observation):**|
| :- |
|`    `task\_id: str                     # 'transaction\_dispute' | 'card\_block' | 'loan\_enquiry'|
|`    `turn: int                        # Current turn number (starts at 1)|
|`    `customer\_message: str            # What the customer just said|
|`    `conversation\_history: List[dict] # [{role, content}] full chat so far|
|`    `account\_context: dict            # Account data agent is ALLOWED to see|
|`    `compliance\_flags: List[str]      # Rules broken so far this episode|
|`    `# done: bool  (inherited from Observation base)|
|`    `# reward: Optional[float]  (inherited)|

account\_context is deliberately limited. The agent sees account type, join date, and recent transaction summaries — but NOT full account numbers or card numbers. This tests whether the agent handles partial information correctly.

**4.3  BankSupportState**

|**class BankSupportState(State):**|
| :- |
|`    `task\_id: str|
|`    `scenario: dict               # Full scenario including hidden ground truth|
|`    `identity\_verified: bool = False|
|`    `issue\_identified: bool = False|
|`    `required\_info\_collected: List[str] = []|
|`    `compliance\_violations: List[str] = []|
|`    `# episode\_id, step\_count inherited from State base|

State holds the ground truth that the grader needs but the agent cannot see. For example, the correct transaction amount for Task 1, or all three required clarifying questions for Task 3.

# **5. Task Definitions**
Three tasks with a clear easy→medium→hard difficulty curve. Each task runs for a maximum of 6 turns (multi-turn is required by the scoring rubric for the medium and hard tasks).

## **5.1  Task 1 — Transaction Dispute (Easy)**

|**Task ID: transaction\_dispute**|
| :- |
|Max turns: 3  |  Max score: 1.0  |  Difficulty: Easy|

**Scenario**

A customer contacts support claiming an unrecognised charge of ₹4,500 appeared on their account on a specific date. The agent must verify the customer’s identity, confirm which transaction they’re referring to, and explain the dispute process.

**Why this is ‘easy’**

- Single, clear issue with an obvious resolution path
- Identity verification is a binary check — either asked or not
- Dispute process explanation can be graded with keyword matching
- No multi-turn complexity, no ambiguity

**Grader breakdown**

|**Criterion**|**How graded**|**Weight**|
| :- | :- | :- |
|Identity verified before helping|Check: response contains verification question before any account info|0\.30|
|Correct transaction identified|Check: agent references correct amount/date from account\_context|0\.30|
|Dispute process explained|LLM judge: does response explain filing a dispute? (0–1 scaled to 0.25)|0\.25|
|Professional tone, no data leaked|Check: no full account numbers in response, polite language|0\.15|

## **5.2  Task 2 — Card Block & Replacement (Medium)**

|**Task ID: card\_block**|
| :- |
|Max turns: 5  |  Max score: 1.0  |  Difficulty: Medium|

**Scenario**

A customer reports their debit card lost or stolen. The agent must: verify identity, confirm the card will be blocked, collect a delivery address for the replacement card, give an expected timeline, and then handle a follow-up question about whether the new card will work internationally.

**Why this is ‘medium’**

- Multi-turn: the follow-up question comes in a second customer message
- Agent must both take an action (block card) and collect information (address)
- The international usage question tests whether the agent knows product details
- Requires maintaining context across turns: agent can’t ask for identity twice

**Grader breakdown**

|**Criterion**|**How graded**|**Weight**|
| :- | :- | :- |
|Identity verified before card action|Check: verification question appears before block confirmation|0\.25|
|Card block confirmed|Check: response contains confirmation language + block-related keywords|0\.20|
|Delivery address collected|Check: agent asked for delivery address in one of its turns|0\.20|
|International usage answered|LLM judge: is the international usage answer accurate and complete?|0\.20|
|Overall resolution quality|LLM judge: holistic quality score across full conversation|0\.15|

## **5.3  Task 3 — Loan Eligibility Enquiry (Hard)**

|**Task ID: loan\_enquiry**|
| :- |
|Max turns: 6  |  Max score: 1.0  |  Difficulty: Hard  |  Penalty: -0.30|

**Scenario**

A customer asks about taking a personal loan of ₹2,50,000. The agent must ask the right clarifying questions (income, employment status, existing debts), give accurate eligibility information without making promises, and recommend escalation to a human advisor if complexity warrants it.

**Why this is ‘hard’**

- Agent must proactively gather information, not just respond to explicit queries
- There is a trap: customer may ask ‘will I definitely get the loan?’ and the agent must NOT say yes
- Escalation decision requires judgment, not rule-following
- LLM judge is used heavily — ground truth is partially subjective

**Grader breakdown**

|**Criterion**|**How graded**|**Weight**|
| :- | :- | :- |
|All 3 clarifying questions asked|Check: income question, employment question, existing debts question all present|0\.30|
|No false guarantees made|Check: response does NOT contain guarantee phrases. Deterministic keyword list.|0\.20|
|Eligibility criteria described accurately|LLM judge: are the criteria described factually correct per scenario?|0\.25|
|Escalation decision appropriate|LLM judge: given scenario complexity, was escalation recommended correctly?|0\.15|
|PENALTY: Made a guarantee|Programmatic: if guarantee phrase detected, subtract 0.30 from total|-0.30|

# **6. Reward Function Design**
This is the most critically evaluated part of the submission. The spec explicitly requires: rewards signal partial progress over the full trajectory (not just binary end-of-episode), and must penalise clearly undesirable behaviour.

**6.1  Partial progress principle**

Reward is computed at every step, not just at episode end. This means the agent gets a signal after each turn, enabling learning. The reward is cumulative but each step can contribute independently.

|**Per-step reward logic:**|
| :- |
|`  `step 1 reward = identity\_verification\_score (0 or 0.30)|
|`  `step 2 reward = issue\_identification\_score + partial\_resolution\_score|
|`  `step 3+ reward = remaining\_criteria\_scores + llm\_judge\_score|
|`  `any step reward -= compliance\_violation\_penalty (if triggered)|
||
|`  `Final score = sum(all step rewards), clamped to [0.0, 1.0]|

**6.2  LLM judge implementation**

The LLM judge is called inside graders.py using the same OpenAI client pattern as inference.py. It uses a tight, structured prompt to return a float between 0 and 1.

|**LLM judge prompt template:**|
| :- |
|`  `You are a strict banking compliance evaluator.|
|`  `Customer query: {customer\_message}|
|`  `Agent response: {agent\_response}|
|`  `Criterion: {criterion\_description}|
|`  `Score the response on this criterion from 0.0 to 1.0.|
|`  `Reply with ONLY a number. No explanation.|

The judge is called with temperature=0 for determinism. Output is parsed as float, clamped to [0, 1], then scaled to the criterion’s weight.

**6.3  Penalty mechanic (Task 3)**

The guarantee penalty is a first-class reward design feature, not just a rule. It demonstrates that the reward function can penalise clearly undesirable behaviour as required by the spec. Implementation uses a keyword list:

|**GUARANTEE\_PHRASES = [**|
| :- |
|`    `'you will definitely get', 'guaranteed approval',|
|`    `'100% approved', 'we will approve', 'you are approved',|
|`    `'i can guarantee', 'no problem getting'|
|]|
|if any(phrase in response.lower() for phrase in GUARANTEE\_PHRASES):|
|`    `reward -= 0.30|
|`    `state.compliance\_violations.append('false\_guarantee')|

# **7. Server Implementation (server/)**
## **7.1  environment.py — Core Logic**
Implements the OpenEnv Environment base class. Three methods to implement: reset(), step(), and the state property.

|**class BankSupportEnvironment(Environment):**|
| :- |
|`    `SUPPORTS\_CONCURRENT\_SESSIONS = True|
||
|`    `def reset(self, task\_id=None, seed=None, \*\*kwargs) -> BankSupportObservation:|
|`        `# 1. Load scenario from tasks.py based on task\_id|
|`        `# 2. Initialise state: identity\_verified=False, violations=[]|
|`        `# 3. Return first customer message as Observation|
||
|`    `def step(self, action: BankSupportAction, \*\*kwargs) -> BankSupportObservation:|
|`        `# 1. Run graders.py on the action|
|`        `# 2. Update state (set flags, add to history)|
|`        `# 3. Generate next customer message (or end episode)|
|`        `# 4. Return Observation with reward and done flag|
||
|`    `@property|
|`    `def state(self) -> BankSupportState:  return self.\_state|

## **7.2  graders.py — Reward Logic**
All scoring logic lives here. Two types of functions:

- Programmatic graders: pure Python, deterministic, fast, no API calls
- LLM graders: call OpenAI client, return float 0–1, used for qualitative criteria

|**# Programmatic grader example**|
| :- |
|def check\_identity\_verified(response: str, history: list) -> float:|
|`    `VERIFY\_PHRASES = ['date of birth', 'registered mobile',|
|`                      `'account number', 'full name', 'verify your']|
|`    `if history:  # Not first turn|
|`        `return 1.0  # Already verified in earlier turn|
|`    `asked = any(p in response.lower() for p in VERIFY\_PHRASES)|
|`    `return 1.0 if asked else 0.0|
||
|# LLM grader example|
|def llm\_score\_resolution\_quality(customer\_msg: str, agent\_response: str) -> float:|
|`    `prompt = JUDGE\_PROMPT.format(customer=customer\_msg, agent=agent\_response)|
|`    `result = openai\_client.chat.completions.create(|
|`        `model=MODEL\_NAME, messages=[{role:'user', content:prompt}],|
|`        `temperature=0, max\_tokens=5)|
|`    `return float(result.choices[0].message.content.strip())|

## **7.3  tasks.py — Scenario Data**
Hardcoded scenario data for all three tasks. Each scenario includes: the opening customer message, follow-up messages for multi-turn tasks, the hidden ground truth used by graders, and account context given to the agent.

|**SCENARIOS = {**|
| :- |
|`  `'transaction\_dispute': {|
|`    `'opening\_message': 'Hi, I see a charge of Rs.4500 from 3rd April'|
|`                      `' that I don’t recognise. Can you help?',|
|`    `'account\_context': { 'account\_type': 'Savings', 'join\_date': '2021-03' },|
|`    `'ground\_truth': { 'disputed\_amount': 4500, 'disputed\_date': '2024-04-03',|
|`                     `'merchant': 'UNKNOWN\_MERCHANT\_XYZ' },|
|`    `'max\_turns': 3|
|`  `},|
|`  `# card\_block and loan\_enquiry follow same structure...|
|}|

## **7.4  app.py — FastAPI Server**

|**from openenv.core.env\_server import create\_fastapi\_app**|
| :- |
|from environment import BankSupportEnvironment|
||
|app = create\_fastapi\_app(BankSupportEnvironment)|
|# Auto-creates: /ws /reset /step /state /health /docs|

No custom routes needed. create\_fastapi\_app() from openenv-core handles all endpoints including the /reset endpoint that the validation script pings to check if the HF Space is live.

# **8. Inference Script (inference.py)**
This is a mandatory file, placed in the root directory. It must use the OpenAI client, read credentials from environment variables, and emit exactly the [START]/[STEP]/[END] stdout format. Any deviation in field names, ordering, or formatting causes incorrect scoring.

**8.1  Environment variables**

|**Variable**|**Description**|
| :- | :- |
|**API\_BASE\_URL**|LLM endpoint. Default: https://router.huggingface.co/v1|
|**MODEL\_NAME**|Model identifier. Default: Qwen/Qwen2.5-72B-Instruct|
|**HF\_TOKEN**|Hugging Face API key. No default — must be set.|
|**LOCAL\_IMAGE\_NAME**|Docker image name if using from\_docker\_image()|

**8.2  Required stdout format**

|**# One line at episode start:**|
| :- |
|[START] task=transaction\_dispute env=bank\_support model=Qwen2.5-72B|
||
|**# One line per step, immediately after env.step() returns:**|
|[STEP] step=1 action=<agent\_response\_text> reward=0.30 done=false error=null|
|[STEP] step=2 action=<agent\_response\_text> reward=0.55 done=false error=null|
|[STEP] step=3 action=<agent\_response\_text> reward=1.00 done=true error=null|
||
|**# One line after env.close(), always emitted even on exception:**|
|[END] success=true steps=3 score=0.85 rewards=0.30,0.55,1.00|

**8.3  Script structure**

1. Import BankSupportEnv client and OpenAI client
1. Read API\_BASE\_URL, MODEL\_NAME, HF\_TOKEN from os.getenv()
1. Run all 3 tasks in sequence (or task specified by env var)
1. For each task: call env.reset(), loop env.step() until done
1. Log [START] once, [STEP] after each step, [END] after env.close()
1. Compute score = sum(rewards) / max\_possible\_reward, clamp to [0,1]
1. Success = score >= 0.5 threshold

Infra constraint: entire inference script must complete in under 20 minutes on vcpu=2, memory=8gb. With 3 tasks, max 6 turns each, and LLM judge calls, this is achievable with Qwen2.5-72B via HF router.

# **9. Client Implementation (client.py)**
The client is what external users import to interact with the environment. It translates between typed Pydantic models and the WebSocket wire format. Three methods to implement following Module 4 pattern:

|**class BankSupportEnv(EnvClient[BankSupportAction, BankSupportObservation, BankSupportState]):**|
| :- |
||
|`    `def \_step\_payload(self, action: BankSupportAction) -> dict:|
|`        `return {'agent\_response': action.agent\_response}|
||
|`    `def \_parse\_result(self, payload: dict) -> StepResult:|
|`        `obs\_data = payload.get('observation', {})|
|`        `return StepResult(|
|`            `observation=BankSupportObservation(|
|`                `task\_id=obs\_data.get('task\_id'),|
|`                `turn=obs\_data.get('turn', 1),|
|`                `customer\_message=obs\_data.get('customer\_message', ''),|
|`                `conversation\_history=obs\_data.get('conversation\_history', []),|
|`                `account\_context=obs\_data.get('account\_context', {}),|
|`                `compliance\_flags=obs\_data.get('compliance\_flags', []),|
|`                `done=payload.get('done', False),|
|`                `reward=payload.get('reward'),|
|`            `),|
|`            `reward=payload.get('reward'),|
|`            `done=payload.get('done', False),|
|`        `)|
||
|`    `def \_parse\_state(self, payload: dict) -> BankSupportState:|
|`        `return BankSupportState(\*\*payload)|

# **10. openenv.yaml**
This manifest file is validated by openenv validate. It must declare the environment metadata, list all tasks with their graders, and reference the correct class paths.

|**name: bank-support-env**|
| :- |
|version: 1.0.0|
|description: Multi-turn banking customer support RL environment|
|tags: [finance, customer-support, multi-turn, compliance, openenv]|
|author: <your-hf-username>|
||
|environment:|
|`  `class: server.environment:BankSupportEnvironment|
|`  `client: client:BankSupportEnv|
||
|tasks:|
|`  `- id: transaction\_dispute|
|`    `name: Transaction Dispute Resolution|
|`    `difficulty: easy|
|`    `max\_steps: 3|
|`    `reward\_range: [0.0, 1.0]|
|`  `- id: card\_block|
|`    `name: Card Block and Replacement|
|`    `difficulty: medium|
|`    `max\_steps: 5|
|`    `reward\_range: [0.0, 1.0]|
|`  `- id: loan\_enquiry|
|`    `name: Loan Eligibility Enquiry|
|`    `difficulty: hard|
|`    `max\_steps: 6|
|`    `reward\_range: [-0.3, 1.0]|

# **11. Dockerfile**
Must live in server/. The validation script checks both server/Dockerfile and root Dockerfile. docker build && docker run must work cleanly.

|**FROM python:3.11-slim**|
| :- |
||
|WORKDIR /app|
|COPY requirements.txt .|
|RUN pip install --no-cache-dir -r requirements.txt|
||
|COPY . .|
||
|EXPOSE 8000|
|CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]|
||
|# requirements.txt:|
|# openenv-core|
|# fastapi|
|# uvicorn|
|# pydantic>=2.0|
|# openai|
|# python-dotenv|

# **12. Build Order & Development Workflow**
Build in this exact sequence. Each step is a working checkpoint you can test before moving on.

|**#**|**File**|**What to do & how to test**|
| :- | :- | :- |
|Step 1|models.py|Define Action, Observation, State Pydantic classes. Run: python models.py. If it imports without error, done.|
|Step 2|server/tasks.py|Write 3 scenario dicts with hardcoded customer messages and ground truth. No logic, just data.|
|Step 3|server/graders.py (programmatic only)|Write all keyword-based checks. Test each function independently with handcrafted inputs.|
|Step 4|server/environment.py|Wire reset() and step() using tasks.py and graders.py. Test with a manual Python loop (no server).|
|Step 5|server/graders.py (LLM judge)|Add LLM judge functions. Test with real API call using HF\_TOKEN.|
|Step 6|server/app.py|One line: create\_fastapi\_app(). Run uvicorn locally, hit /reset with curl.|
|Step 7|client.py|Implement \_step\_payload and \_parse\_result. Test with local server running.|
|Step 8|inference.py|Write baseline script. Run it. Verify [START]/[STEP]/[END] output matches spec exactly.|
|Step 9|openenv.yaml|Write manifest. Run: openenv validate. Fix any errors.|
|Step 10|Dockerfile|Build image. Run container. Hit /health endpoint from outside.|
|Step 11|HF Space|Push to Hugging Face. Run validate-submission.sh against your Space URL.|
|Step 12|README.md|Document everything: action/observation spaces, task descriptions, setup, baseline scores.|

# **13. Pre-Submission Validation Checklist**
The validation script checks 3 things automatically. All must pass or you are disqualified. Run this before submitting:

|**curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh \**|
| :- |
|`  `| bash -s -- https://<your-space>.hf.space|

|**Validation check**|**What it verifies**|
| :- | :- |
|**Step 1: HF Space live**|POST /reset returns HTTP 200. Space must be running and not sleeping.|
|**Step 2: Docker build**|docker build completes within 600s timeout. No build errors.|
|**Step 3: openenv validate**|Manifest, typed models, all endpoints validated by openenv-core.|

**Manual checks (not automated but judged)**

- inference.py runs end-to-end and produces scores for all 3 tasks
- Graders return different scores for different inputs (not always same value)
- Reward values in [0.0, 1.0] range (or [-0.30, 1.0] for Task 3 with penalty)
- README covers all 5 required sections: description, spaces, tasks, setup, baseline scores
- Runtime under 20 minutes on vcpu=2, memory=8gb

# **14. README.md Required Sections**
The README is explicitly required by the spec. It must contain all five of these sections to pass human review.

|**Section**|**What to include**|
| :- | :- |
|**Environment description & motivation**|What BankSupportEnv does, why banking support is a valuable domain, what makes it a good RL environment.|
|**Action & observation space definitions**|Exact field names and types from models.py. What the agent sees, what it can do.|
|**Task descriptions with difficulty**|All 3 tasks: what the customer wants, what the agent must do, why it’s easy/medium/hard.|
|**Setup & usage instructions**|pip install, env variables to set, how to run the server, how to run inference.py.|
|**Baseline scores**|Actual scores from running inference.py with Qwen2.5-72B. One score per task.|

# **15. Risk Register**

|**Risk**|**Severity**|**Mitigation**|
| :- | :- | :- |
|LLM judge API call fails during grading|High|Wrap in try/except. Return 0.5 (neutral score) on failure. Log error to compliance\_flags.|
|HF Space goes to sleep before validation ping|Medium|Keep Space on paid tier or ensure it was pinged recently. Test /reset manually before submitting.|
|Graders return same score for all inputs|High (disqualifies)|Test graders with at least 5 diverse inputs: empty response, perfect response, wrong response, partial response, violation response.|
|inference.py stdout format wrong|High (disqualifies)|Copy log functions verbatim from sample inference.py. Do not modify field names or ordering.|
|Docker build fails on HF infra|Medium|Use python:3.11-slim base. Avoid heavy ML dependencies. Keep requirements minimal.|
|Step count exceeds 20 minute limit|Medium|Profile with 3 tasks × 6 turns × 2 LLM calls per step. If tight, reduce max\_turns or cache judge calls.|

# **16. Key Terms Glossary**

|**Term**|**Definition**|
| :- | :- |
|**OpenEnv**|Open-source framework by Meta & Hugging Face for standardised RL environments.|
|**Gymnasium**|Python API standard (reset/step/render) that OpenEnv builds on.|
|**Observation**|What the agent sees each turn. A Pydantic model returned by reset() and step().|
|**Action**|What the agent does. A Pydantic model passed to step().|
|**State**|Internal environment state. Hidden from agent, visible to graders.|
|**Episode**|One full run from reset() to done=True.|
|**Grader**|Function that scores an agent’s response. Returns float in [0.0, 1.0].|
|**LLM-as-judge**|Using an LLM to score qualitative aspects of a response. Called inside graders.py.|
|**Partial reward**|Reward given at each step, not just at episode end. Required by spec.|
|**HF Space**|Hugging Face-hosted container. Where the environment server runs publicly.|
|**openenv validate**|CLI command that checks spec compliance: models, endpoints, manifest.|
|**inference.py**|Mandatory baseline script. Must be in root. Uses OpenAI client. Structured stdout.|

