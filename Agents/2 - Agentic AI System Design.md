# AI Systems Design & Architecture — Deployment, Integration, Monitoring & Guardrails

## Chapter 1: From Code to System — Architecture Mental Models

### 1.1 What changes between "an agent that works" and "a system in production"

Simple explanation: a working Python script is like a car engine sitting on a workbench — it runs, but a real car needs a chassis, brakes, fuel lines, and a dashboard around it before anyone can drive it safely. This chapter covers the chassis, not the engine.

Technical framing: the previous manual covered the engine — models, chains, RAG, agents, LangGraph. This one covers everything that turns that engine into a system other people and other services can depend on: deployment, scaling, integration boundaries, safety guardrails, observability, and cost control.

### 1.2 Stateless service, external state

Simple explanation: the API server should be like a waiter who takes your order and forgets you the moment you leave — all the "remembering" happens in the kitchen's notebook (the database), not in the waiter's head.

Technical reasoning: any API instance should be able to crash and restart, or be one of ten identical replicas behind a load balancer, without losing conversation state. This is only possible if state (chat history, checkpoints, long-term memory) lives in an external store (MongoDB, Redis) rather than in-process memory.

```
        Load Balancer
       /      |       \
   API-1    API-2    API-3      <- stateless, interchangeable, disposable
       \      |       /
      Shared state store (MongoDB / Redis)
```

### 1.3 Sync vs async request handling

| Model | Behavior | When to use |
|---|---|---|
| Synchronous request/response | Client waits, connection stays open until the full answer is ready | Simple single-turn queries, short agent runs |
| Streaming (SSE/WebSocket) | Client gets tokens/events as they're produced | Chat UIs, anything where perceived latency matters |
| Async job + polling | Client gets a job id immediately, polls or gets a webhook later | Long-running multi-agent tasks, batch processing |

### 1.4 Full system architecture — the map for the rest of this document

```
   Client (web/mobile)
         |
   API Gateway / Load Balancer
         |
   Application layer (FastAPI + LangGraph agent)  <-- guardrails wrap this layer
         |         |              |
   State store   Vector DB     External tools/MCP servers
   (Mongo/Redis) (RAG index)   (internal APIs, third-party services)
         |
   Observability (logs, metrics, traces) -- cuts across every layer above
```

Every chapter below fills in one part of this diagram: deployment (how the application layer gets shipped and scaled), integration (the arrows between boxes), guardrails (the wrapper around the application layer), and monitoring (the layer that watches everything).

## Chapter 2: Deployment Fundamentals

### 2.1 Containerizing the application

Simple explanation: a container is a shipping crate — whatever runs correctly on your laptop inside that crate will run identically on any machine that can accept the crate, regardless of what's installed on that machine.

General syntax:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["<start command>"]
```

Real example — Dockerfile for the FastAPI + LangGraph agent from the previous manual:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 2.2 Configuration and secrets

Simple explanation: never hardcode a password into a file the way you'd never write your bank PIN on a sticky note on your monitor — configuration that differs by environment or that is sensitive belongs outside the code.

| What | Where it lives | Example |
|---|---|---|
| Non-sensitive config (model name, timeouts) | Environment variables, config file per environment | `MODEL_NAME=gpt-4o-mini` |
| Secrets (API keys, DB passwords) | Secret manager (AWS Secrets Manager, GCP Secret Manager, Vault) or injected env vars at deploy time, never committed to git | `OPENAI_API_KEY` |

```python
import os

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]   # fails loudly if missing, which is correct
```

### 2.3 Process management

Simple explanation: `uvicorn --workers 4` is like opening four checkout counters at a store instead of one — more customers (requests) can be served at the same time.

```
gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 api:app
```

Gunicorn manages worker processes (restarting crashed ones, distributing load); Uvicorn workers handle the actual async request execution underneath it. This combination is the standard way to run a FastAPI app in production rather than the bare `uvicorn --reload` used in development.

### 2.4 Deployment targets

| Target | Fit |
|---|---|
| Single VM | Small internal tools, early-stage projects |
| Container orchestration (Kubernetes, ECS) | Multiple services, need for autoscaling, rolling deploys |
| Serverless (AWS Lambda, Google Cloud Run) | Spiky/unpredictable traffic, willing to accept cold-start latency |
| Managed agent platforms (LangGraph Platform) | Want built-in checkpointing, streaming, and deployment without building the orchestration infrastructure yourself |

### 2.5 Rolling and blue-green deployments

Simple explanation: rolling deployment is like replacing tires on a moving car one at a time; blue-green is like having two identical cars and switching which one the driver uses, so you can switch back instantly if something's wrong.

| Strategy | How it works | Rollback speed |
|---|---|---|
| Rolling | Old instances replaced by new ones gradually, a few at a time | Moderate — needs to roll back gradually too |
| Blue-green | Full new environment ("green") built alongside the old ("blue"); traffic is switched over at once | Instant — switch traffic back to blue |
| Canary | A small percentage of traffic goes to the new version first, increased gradually if healthy | Fast — cut canary traffic to zero |

For agentic systems specifically, canary releases matter more than for typical web apps because a new prompt or model version can silently degrade answer quality without throwing any errors — this is covered further in the CI/CD chapter.

## Chapter 3: Scaling & Performance

### 3.1 The concurrency model for LLM-bound workloads

Simple explanation: waiting for an LLM response is like waiting for a kettle to boil — you don't stand frozen staring at it, you go do something else and come back when it's ready. Async code lets one process handle many such "kettles" at once instead of one at a time.

Technical reasoning: LLM and tool calls are I/O-bound (waiting on network), not CPU-bound. `async def` endpoints with `await model.ainvoke(...)` let a single worker process handle many concurrent requests instead of blocking on each one.

```python
@app.post("/chat")
async def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.user_id}}
    result = await graph_app.ainvoke({"messages": [HumanMessage(req.message)]}, config=config)
    return {"reply": result["messages"][-1].content}
```

### 3.2 Connection pooling

Every database and vector store client should reuse a small pool of connections rather than opening a new one per request — opening a fresh MongoDB connection on every chat message is the same mistake as opening a new browser tab for every single Google search.

```python
from pymongo import MongoClient

mongo_client = MongoClient("mongodb://localhost:27017", maxPoolSize=50)   # created once at startup, reused everywhere
```

### 3.3 Caching layers

| Cache type | What it stores | Saves |
|---|---|---|
| Exact-match response cache | Identical (prompt, params) pairs mapped to a stored response | Repeated identical queries |
| Semantic cache | Embeddings of past queries, matched by similarity rather than exact text | Rephrased but equivalent queries ("refund policy" vs "how do refunds work") |
| Prompt/context caching (provider-level) | The provider caches a long, repeated system prompt or document context server-side | Cost/latency on large repeated context blocks |

```python
import hashlib

cache = {}

def cached_invoke(prompt: str):
    key = hashlib.sha256(prompt.encode()).hexdigest()
    if key in cache:
        return cache[key]
    result = model.invoke(prompt)
    cache[key] = result
    return result
```

A semantic cache follows the same idea but looks up the nearest neighbor in a small vector store of past queries instead of an exact dictionary key, with a similarity threshold before returning a cached hit.

### 3.4 Model fallback and load balancing across providers

Simple explanation: keep a backup generator running, the same way a hospital does, so a power (provider) outage doesn't take the whole system down.

```python
def invoke_with_fallback(prompt, primary, fallback):
    try:
        return primary.invoke(prompt)
    except Exception:
        return fallback.invoke(prompt)

response = invoke_with_fallback(prompt, model_primary_openai, model_fallback_anthropic)
```

### 3.5 Background workers for long-running agent tasks

Simple explanation: some agent tasks are like sending a package for delivery — the client doesn't need to wait at the door for hours, they get a tracking number and check back later.

General pattern using a task queue (Celery, RQ, or a cloud equivalent):

```python
@app.post("/tasks")
async def start_task(req: TaskRequest):
    job = queue.enqueue(run_agent_task, req.user_id, req.instructions)
    return {"job_id": job.id}

@app.get("/tasks/{job_id}")
async def get_task_status(job_id: str):
    job = queue.fetch_job(job_id)
    return {"status": job.get_status(), "result": job.result}
```

The API layer stays fast and stateless; the actual multi-step agent run happens in a separate worker process, with its state and result tracked through the queue's job store.

### 3.6 Horizontal scaling recap

Because state lives externally (Chapter 1.2) and connections are pooled (3.2), the application layer can be scaled by simply running more identical instances behind a load balancer — this is the entire point of designing it stateless from the start.

## Chapter 4: Integration Patterns

### 4.1 API-first design

Simple explanation: design the "door" into your agent system before designing the rooms behind it — every other team or frontend only ever interacts through that door, so it needs to be stable even as the internals change.

```python
class ChatRequest(BaseModel):
    user_id: str
    message: str
    metadata: dict | None = None   # room to extend later without breaking existing clients

class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    tool_calls_made: list[str] = []
```

Returning `tool_calls_made` and similar metadata alongside the reply gives consuming teams visibility without them needing access to your logs or traces.

### 4.2 Streaming to a frontend (SSE)

Simple explanation: Server-Sent Events are like a sports commentary feed — the server keeps talking, and the client just listens for each new update instead of asking "is it done yet?" repeatedly.

```python
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        config = {"configurable": {"thread_id": req.user_id}}
        async for event in graph_app.astream(
            {"messages": [HumanMessage(req.message)]}, config=config, stream_mode="messages"
        ):
            yield f"data: {event[0].content}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

Client side:

```javascript
const eventSource = new EventSource("/chat/stream");
eventSource.onmessage = (e) => appendToChatUI(e.data);
```

### 4.3 Async job pattern for long tasks (recap from performance, viewed as an integration contract)

| Endpoint | Purpose |
|---|---|
| `POST /tasks` | Submit work, returns a `job_id` immediately |
| `GET /tasks/{job_id}` | Poll for status/result |
| Webhook callback (optional) | Server calls back to a client-provided URL when done, instead of the client polling |

### 4.4 Event-driven integration with existing microservices

Simple explanation: instead of one service directly calling another and waiting, both services talk through a shared mailbox (a message queue) — the sender drops a message and moves on, the receiver picks it up whenever it's ready.

```
Order Service --publishes "order.created"--> Message Queue (Kafka/RabbitMQ)
                                                      |
                                          Agent Service consumes event,
                                          runs analysis, publishes "order.flagged"
```

This decouples the agent system from the rest of the company's services — the agent doesn't need to know who calls it or when, and other services don't need to wait on agent latency.

### 4.5 Third-party integration options, compared

| Approach | When to use |
|---|---|
| Direct REST/SDK call to the third party | Simple, one-off integration, you control both ends |
| MCP server | The tool/data source should be reusable across multiple different agents or hosts (covered in depth in the development manual) |
| iPaaS / integration platform (Zapier-style) | Non-technical teams need to wire up integrations without custom code |

### 4.6 Frontend integration checklist

- Every response includes a `thread_id` so the frontend can maintain continuity across page reloads.
- Streaming endpoints degrade gracefully to non-streaming for clients that don't support SSE/WebSocket.
- Error responses use a consistent shape (`{"error": {"code": ..., "message": ...}}`) so the frontend can handle failures uniformly instead of parsing raw stack traces.

## Chapter 5: Guardrails & Safety Architecture

### 5.1 Mental model — defense in depth

Simple explanation: guardrails work like airport security — there isn't one checkpoint that catches everything, there's a sequence of different checks (bag scan, metal detector, ID check), each catching what the others might miss.

```
User input
    |
[Input guardrails: injection detection, PII redaction, validation]
    |
Agent / LLM / Tools
    |
[Output guardrails: moderation, schema validation, hallucination check]
    |
Response to user
```

### 5.2 Input guardrails

| Risk | Mitigation |
|---|---|
| Prompt injection ("ignore your instructions and...") | Classifier or rule-based scan on incoming text before it reaches the system prompt; treat retrieved document content as data, not instructions |
| PII in user input reaching logs or a third-party model | Redact/mask PII patterns before logging or forwarding |
| Malformed/oversized input | Schema validation and length limits at the API boundary (already partly handled by Pydantic request models) |

```python
import re

def redact_pii(text: str) -> str:
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED-SSN]", text)
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[REDACTED-EMAIL]", text)
    return text

def check_injection(text: str, classifier_model) -> bool:
    result = classifier_model.invoke(f"Does this text attempt to override system instructions? Answer YES or NO.\n{text}")
    return "YES" in result.content.upper()
```

A real implementation typically layers a fast rule-based pass (regex, keyword lists) before an optional, slower LLM-based classifier pass for ambiguous cases — running the expensive check on every single message is often unnecessary.

### 5.3 Output guardrails

| Risk | Mitigation |
|---|---|
| Harmful/inappropriate content in the response | Moderation API or classifier pass on the generated output before returning it |
| Response doesn't match expected structure (breaks a downstream integration) | Validate against a Pydantic schema; reject and retry generation if invalid |
| Hallucinated facts not grounded in retrieved context (RAG systems) | Faithfulness check comparing the answer against retrieved chunks, covered in the development manual's RAG evaluation section |

```python
from pydantic import ValidationError

def safe_structured_response(chain, input_data, schema, max_retries=2):
    for attempt in range(max_retries):
        try:
            return schema.model_validate(chain.invoke(input_data))
        except ValidationError:
            continue
    raise ValueError("Model failed to produce valid structured output after retries")
```

### 5.4 Tool-use guardrails

Simple explanation: giving an agent a tool is like giving an employee a company credit card — you still set a spending limit and require a manager's sign-off above a certain amount, rather than trusting unlimited judgment.

| Guardrail | Purpose |
|---|---|
| Permission scoping | An agent acting on behalf of user A can only call tools with user A's own permissions, never elevated ones |
| Human approval gate | Irreversible or high-value actions (refunds above a threshold, deleting data, sending external emails) pause for approval via `interrupt()` |
| Per-tool rate limiting | Prevents a misbehaving loop from hammering an internal API or running up cost |
| Argument validation | Tool inputs are validated (e.g. an order ID must match an expected format) before execution, not trusted blindly from model output |

```python
@tool
def process_refund(order_id: str, amount: float) -> str:
    """Process a refund for an order."""
    if amount > 100:
        return "This refund exceeds the auto-approval limit and requires human approval."
    return execute_refund(order_id, amount)
```

Combined with the `interrupt()` pattern from the development manual, amounts above the threshold would route to a human-approval node instead of returning this message directly.

### 5.5 Dedicated guardrail frameworks

| Framework | Focus |
|---|---|
| Guardrails AI | Schema-based validation and structured re-asking when output fails checks |
| NeMo Guardrails | Conversational flow rules, topic restriction, jailbreak resistance |
| Custom LLM-as-judge classifiers | Flexible, tailored checks (tone, policy compliance) using a smaller/cheaper model as the judge |

These are worth adopting once guardrail logic outgrows a handful of regex and if-statements — they provide tested rule sets so you're not reinventing prompt-injection detection from scratch.

## Chapter 6: Observability & Monitoring

### 6.1 The three pillars, applied to agentic systems

Simple explanation: logs are a diary of what happened, metrics are the dashboard gauges showing current health, and traces are the GPS route showing exactly which path a specific request took through the system.

| Pillar | What it captures | Example for an agent |
|---|---|---|
| Logs | Discrete events, human-readable | "Tool `check_order_status` failed for order A123: timeout" |
| Metrics | Aggregated numbers over time | p95 latency, requests per minute, error rate |
| Traces | The full path of one specific request through every node/tool | supervisor -> billing agent -> `calculate_refund` tool -> response |

### 6.2 LLM/agent-specific metrics worth tracking

| Metric | Why it matters |
|---|---|
| Tokens in / tokens out per request | Direct cost driver |
| Latency per node/tool call | Finds the actual bottleneck instead of guessing |
| Tool-call success/failure rate | Surfaces flaky external dependencies |
| Number of agent loop iterations per request | Detects runaway loops before they hit the recursion limit and waste cost |
| Fallback/retry rate | Signals a degrading primary provider |

### 6.3 Tracing agent execution

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "..."
os.environ["LANGSMITH_PROJECT"] = "support-agent-prod"
```

For systems not using LangSmith, the same idea is achieved with OpenTelemetry spans wrapping each node/tool call, exported to any standard tracing backend (Jaeger, Datadog, Honeycomb):

```python
from opentelemetry import trace

tracer = trace.get_tracer("agent-service")

def traced_tool_call(tool_fn, *args, **kwargs):
    with tracer.start_as_current_span(tool_fn.name):
        return tool_fn.invoke(*args, **kwargs)
```

### 6.4 Dashboards and alerting

Simple explanation: a dashboard without alerts is like a smoke detector with no battery — it's technically present but won't actually wake anyone up when something's wrong.

| Alert | Trigger |
|---|---|
| Error rate spike | Error rate exceeds a threshold over a rolling window |
| Latency SLO breach | p95 latency exceeds an agreed target for a sustained period |
| Cost anomaly | Token spend in a time window exceeds historical baseline by a set margin |
| Tool failure spike | A specific tool's failure rate crosses a threshold, often indicating an upstream outage |

### 6.5 Evaluation in production

Simple explanation: monitoring tells you the system is running; evaluation tells you whether it's still giving good answers — a server can have 0% errors while quietly giving wrong answers to every user.

| Technique | What it does |
|---|---|
| Online evaluation | A sample of real production responses is scored automatically (LLM-as-judge or rule-based) on an ongoing basis |
| Human feedback loop | Thumbs up/down or corrections from real users feed back into a labeled dataset for future regression testing |
| A/B testing | Two prompt/model versions are served to different user segments, compared on quality and business metrics before a full rollout |

This closes the loop back into the development manual's RAG/agent evaluation section — the same metrics (faithfulness, relevance, tool accuracy) that were checked offline before launch should keep being sampled and checked after launch, not just once before shipping.

## Chapter 7: Reliability Engineering for Agentic Systems

### 7.1 Timeouts and retries with backoff

Simple explanation: if a friend doesn't answer a call, you don't call again instantly ten times in a row — you wait a bit, try again, and eventually give up and try a different way to reach them.

```python
import time

def call_with_retry(fn, *args, max_retries=3, base_delay=1.0, **kwargs):
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))   # exponential backoff
```

Every external call inside a tool or model wrapper (LLM API, database, third-party tool) should be wrapped this way rather than left to fail on the first hiccup.

### 7.2 Circuit breakers

Simple explanation: a circuit breaker in your house trips to stop feeding power into a fault instead of letting it burn continuously — a software circuit breaker stops sending requests to a dependency that's clearly already failing, instead of retrying into a wall.

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=30):
        self.failures = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.open_until = 0

    def call(self, fn, *args, **kwargs):
        if time.time() < self.open_until:
            raise RuntimeError("Circuit open, skipping call")
        try:
            result = fn(*args, **kwargs)
            self.failures = 0
            return result
        except Exception:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.open_until = time.time() + self.reset_timeout
            raise
```

### 7.3 Fallback and graceful degradation

| Failure | Degradation strategy |
|---|---|
| Primary LLM provider down | Fall back to a secondary provider/model (Chapter 3.4) |
| Vector store unreachable | Answer without retrieved context, clearly noting reduced confidence, rather than failing the whole request |
| A non-critical tool fails | Continue the agent run without that tool's result rather than aborting entirely |

### 7.4 Idempotency for tool actions

Simple explanation: pressing an elevator button that's already lit shouldn't call the elevator twice — an idempotent action produces the same real-world result no matter how many times it's accidentally triggered.

This matters specifically because retries (7.1) can cause a tool to be called more than once for what the user thinks was a single request — a refund tool without idempotency protection could double-refund a customer.

```python
@tool
def process_refund(order_id: str, amount: float, idempotency_key: str) -> str:
    """Process a refund for an order, safe to retry with the same idempotency_key."""
    existing = db.refunds.find_one({"idempotency_key": idempotency_key})
    if existing:
        return f"Refund already processed: {existing['status']}"
    result = execute_refund(order_id, amount)
    db.refunds.insert_one({"idempotency_key": idempotency_key, "status": result})
    return result
```

### 7.5 Dead-letter queues for failed async tasks

For the background-worker pattern from Chapter 3.5, a task that fails repeatedly should move to a separate "dead-letter" queue for manual inspection rather than being retried forever or silently dropped:

```
Task fails 3 times --> moved to dead_letter_queue --> alert fired --> engineer inspects manually
```

## Chapter 8: Security Architecture

### 8.1 Secrets and credentials

Covered operationally in Chapter 2.2; architecturally, the rule is that no service — including the agent itself — should have more access than it strictly needs. An agent's database credentials should be scoped to only the collections it actually needs to read or write.

### 8.2 Authentication and authorization

| Layer | Question it answers |
|---|---|
| Authentication | Who is making this request |
| Authorization | What is this specific user/agent allowed to do |
| Tool-level scoping | Even once authenticated, can this user's agent call this specific tool (e.g., only support staff can trigger a refund tool, not any authenticated user) |

```python
@app.post("/chat")
async def chat(req: ChatRequest, current_user: User = Depends(get_current_user)):
    allowed_tools = get_tools_for_role(current_user.role)
    agent = create_react_agent(model, tools=allowed_tools)
    ...
```

The set of tools handed to `create_react_agent` is itself an authorization boundary — an agent literally cannot call a tool it was never given, which makes tool assignment a natural place to enforce role-based permissions.

### 8.3 Data privacy

| Concern | Practice |
|---|---|
| PII in transit | TLS everywhere, including between internal services |
| PII at rest | Encryption at rest for the state store and vector store; consider whether embeddings of sensitive text also need protection |
| Data residency | Some regulations require certain user data to stay within a specific geographic region — this affects which model provider/region and which database region can be used |
| Retention | Chat histories and logs should have a defined retention/deletion policy, not be kept forever by default |

### 8.4 Prompt injection and jailbreak defenses (system-level view)

This was covered as an input guardrail in Chapter 5.2; at the architecture level, the additional principle is: treat any content the agent didn't originate — retrieved documents, tool outputs, third-party API responses — as untrusted data, never as instructions. A malicious instruction embedded inside a retrieved PDF is exactly as dangerous as one typed directly by a user, and should pass through the same input-guardrail layer before being placed in context.

### 8.5 Sandboxing code-execution tools

Simple explanation: if a tool lets the agent run code, that code needs its own locked room to run in — the same way you wouldn't let a stranger run a program directly on your personal laptop.

| Practice | Purpose |
|---|---|
| Run in an isolated container/VM with no network access by default | Limits blast radius if generated code is malicious or buggy |
| Resource limits (CPU, memory, execution time) | Prevents a runaway or infinite-loop script from exhausting shared resources |
| No access to the host filesystem or credentials | Prevents code execution from becoming a path to broader system compromise |

### 8.6 Audit logging

Every tool call that changes state (refunds, deletions, sent messages) should be logged with who/what triggered it, what arguments were used, and what the result was — this is what makes an incident reviewable after the fact and is often a compliance requirement, not just a debugging convenience.

## Chapter 9: Cost Architecture & FinOps for LLM Systems

### 9.1 Token cost modeling

Simple explanation: every LLM call is a metered utility bill, not a flat-rate service — the more text goes in and comes out, the more it costs, so architecture decisions have a direct dollar impact.

```python
def estimate_cost(input_tokens, output_tokens, input_rate, output_rate):
    return (input_tokens / 1000) * input_rate + (output_tokens / 1000) * output_rate
```

Tracking this per request (Chapter 6.2) is what makes the remaining strategies in this chapter measurable rather than guesswork.

### 9.2 Model tiering

Simple explanation: don't send every question to the most expensive expert in the building — have a receptionist (a small, cheap model) triage first, and only escalate to the expensive specialist when the task actually needs it.

```python
router_model = ChatOpenAI(model="gpt-4o-mini")   # cheap, fast, used for routing/classification
answer_model = ChatOpenAI(model="gpt-4o")        # more capable, used only for the final response

intent = router_model.with_structured_output(Intent).invoke(query)
if intent.requires_deep_reasoning:
    response = answer_model.invoke(query)
else:
    response = router_model.invoke(query)
```

### 9.3 Caching to reduce redundant calls

Directly reduces cost as a side effect of the caching strategies already covered in Chapter 3.3 — worth re-checking cache hit rate specifically as a cost metric, not only a latency one.

### 9.4 Budget alerts and hard caps

| Control | Purpose |
|---|---|
| Soft budget alert | Notifies engineering/finance when spend crosses a threshold, before it becomes a problem |
| Hard per-user/per-org cap | Stops serving further requests once a limit is hit, preventing a single misbehaving client or runaway loop from generating an unbounded bill |
| Per-request token limit | `max_tokens` set on every model call, preventing any single response from running unexpectedly long |

```python
model = ChatOpenAI(model="gpt-4o-mini", max_tokens=500)
```

## Chapter 10: CI/CD for Agentic Applications

### 10.1 The testing pyramid, adapted for agents

Simple explanation: the same pyramid used for normal software testing applies here, just with an extra layer added at the top for judging answer quality, which regular software doesn't need because normal code either works or doesn't — an LLM's output has degrees of "good."

```
        Eval-based regression tests   (few, slow, judge answer quality)
        Integration tests on the graph (moderate, test full multi-node flows)
        Unit tests on tools/nodes      (many, fast, deterministic)
```

### 10.2 Unit and integration tests (recap, framed as a pipeline stage)

The node/tool-level tests and full-graph tests already shown in the development manual's testing section belong in this pyramid's bottom two layers and should run on every commit, the same as any other backend test suite.

### 10.3 Eval-based regression tests

Simple explanation: before shipping a new version of a recipe, you have a panel taste-test it against a set of known dishes to make sure it didn't get worse — this is the same idea applied to prompts and models.

```python
eval_dataset = [
    {"input": "What's your refund policy?", "expected_topic": "refund_policy"},
    {"input": "My payment failed twice", "expected_topic": "billing_issue"},
]

def run_eval(agent, dataset):
    results = []
    for case in dataset:
        output = agent.invoke({"messages": [HumanMessage(case["input"])]})
        score = judge_model.invoke(
            f"Does this response address the topic '{case['expected_topic']}'? Answer YES or NO.\n{output}"
        )
        results.append("YES" in score.content.upper())
    return sum(results) / len(results)
```

This is run against every candidate prompt/model change before it's allowed to deploy — a drop in the pass rate blocks the release the same way a failing unit test would.

### 10.4 Prompt and model versioning

| Practice | Purpose |
|---|---|
| Store prompts as versioned files/templates, not inline strings scattered through code | Enables diffing, rollback, and review the same way code changes are reviewed |
| Tag which model version and prompt version produced a given response in logs | Makes it possible to correlate a quality regression with a specific change |
| Rollback plan | Keep the previous prompt/model version deployable within minutes, not requiring a full rebuild |

### 10.5 Canary releases and automated eval gates

```
New version deployed to 5% of traffic
        |
Eval pass rate and error rate monitored for a fixed window
        |
   Pass?  --Yes--> gradually increase to 100%
        |
       No
        |
   Automatic rollback to previous version
```

Wiring the eval suite from 10.3 directly into the deployment pipeline as a gate — rather than running it only manually before a release — is what turns "we think this is fine" into a checked, enforced condition.

## Chapter 11: Capstone — Production Reference Architecture

### 11.1 Full system diagram

```
                        Client (web/mobile)
                                |
                     API Gateway / Load Balancer
                                |
              -------------------------------------
              |                                     |
        FastAPI instance 1                   FastAPI instance N
        (stateless, autoscaled)              (stateless, autoscaled)
              |                                     |
              -------------------------------------
                                |
        --------------------------------------------------
        |              |                |                |
   State store    Vector DB      Background workers   External tools /
   (Mongo/Redis,  (RAG index)    (Celery/RQ, for       MCP servers
   checkpoints,                  long agent tasks)
   long-term memory)
                                |
                  Observability (logs, metrics, traces)
                  -- LangSmith / OpenTelemetry, cuts across every box above
                                |
                  Guardrail layer wraps the application layer:
                  input checks before the agent, output checks after
```

### 11.2 Realistic docker-compose for local/self-hosted setup

```yaml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MONGO_URI=mongodb://mongo:27017
    depends_on:
      - mongo
      - redis
    deploy:
      replicas: 3

  worker:
    build: .
    command: celery -A tasks worker --loglevel=info
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MONGO_URI=mongodb://mongo:27017
    depends_on:
      - redis

  mongo:
    image: mongo:7
    volumes:
      - mongo_data:/data/db

  redis:
    image: redis:7

volumes:
  mongo_data:
```

### 11.3 Consolidated production checklist

| Area | Check |
|---|---|
| Deployment | Containerized, config/secrets externalized, rolling or canary release strategy in place |
| Scaling | Stateless app layer, connection pooling, caching, background workers for long tasks |
| Integration | Consistent API contract, streaming support, event-driven hooks where needed |
| Guardrails | Input and output checks active, tool permissions scoped, human approval on irreversible actions |
| Observability | Logs, metrics, and traces wired up; alerts configured, not just dashboards |
| Reliability | Retries with backoff, circuit breakers, fallback models, idempotent tool actions |
| Security | Secrets externalized, auth/authorization enforced at the tool level, PII handled deliberately, code execution sandboxed |
| Cost | Token usage tracked per request, model tiering in place, hard budget caps set |
| CI/CD | Unit and integration tests automated, eval-based regression tests gating deploys, rollback plan tested |

This checklist is the direct implementation counterpart to the architecture diagram in 11.1 — every box in that diagram maps to at least one row here, and a system isn't production-ready until every row has an actual answer, not just a plan.
