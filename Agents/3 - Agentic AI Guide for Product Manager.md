# Agentic AI — Product & Project Manager's Handbook

## Chapter 1: Vocabulary Every PM Needs

### 1.1 The four terms people conflate

| Term | What it actually means | The PM mistake to avoid |
|---|---|---|
| Predictive AI | Forecasts an outcome from data (churn score, demand forecast) | Calling a dashboard "AI-powered" when it's just predictive analytics |
| Generative AI | Produces new content (text, image, code) from a prompt | Assuming any GenAI feature is autonomous — most are single-turn, not agentic |
| AI Agent | A GenAI model wrapped with tools, memory, and a loop, that takes multi-step action toward a goal | Calling a chatbot with no tools or memory an "agent" |
| Agentic AI | A system of one or more AI agents, often multiple agents coordinating, with enough autonomy to plan and adapt, not just execute a fixed script | Overselling a single scripted agent as "agentic AI" to stakeholders |

### 1.2 Classic agent taxonomy (useful when scoping what you're actually asking engineering to build)

| Type | How it decides | Typical business use |
|---|---|---|
| Simple reflex agent | Fixed rule: condition seen -> action taken, no memory of past or model of the world | Basic chatbot FAQ deflection, simple alerting |
| Model-based reflex agent | Keeps an internal model of the environment's state, decides using rules plus that model | Inventory bots, monitoring agents that track "current state" |
| Goal-based agent | Evaluates possible actions against a defined goal, chooses the one that gets closer to it | Task-completion agents (book this meeting, resolve this ticket) |
| Utility-based agent | Weighs multiple possible actions by how well each achieves the goal, not just whether it does | Agents optimizing across competing priorities (cost vs speed vs satisfaction) |

Most agentic AI products marketed today are goal-based or utility-based agents built on top of an LLM — knowing this taxonomy helps you ask "which kind of decision-making are we actually building" instead of treating every agent as a black box.

### 1.3 Single agent vs multi-agent, from a delivery standpoint

| Factor | Single agent | Multi-agent |
|---|---|---|
| Task shape | One domain, one skill set | Multiple distinct specializations needed (billing + technical + legal) |
| Build/maintenance cost | Lower | Higher — needs an orchestration layer, more testing surface |
| Failure visibility | Easier to trace one agent's decision | Harder — a bad outcome could originate from any agent in the chain |
| When to choose | Default starting point for any new use case | Only once a single agent's tool list or prompt becomes unmanageably broad |

Rule of thumb for scoping: start every initiative as a single-agent pilot. Multi-agent architecture is a scaling decision, not a starting point — greenlighting multi-agent on day one is a common cause of blown timelines.

## Chapter 2: The Business Case — When and Why to Build

### 2.1 When to use an agent, and when not to

| Use an agent when | Do not use an agent when |
|---|---|
| The task is repetitive but has enough variation that fixed rules keep breaking | The task is truly repetitive and rule-based — a script or basic automation is cheaper and more reliable |
| The workflow spans multiple systems/APIs that currently require a human to bridge manually | The decision is high-stakes, irreversible, and low-tolerance for error (without a strong human-in-the-loop gate) |
| Cloud/on-demand scaling of the task matters (volume spikes unpredictably) | The situation is extremely ambiguous or emotionally sensitive with no good fallback if the agent gets it wrong |
| You need to augment a human team's throughput, not replace judgment entirely | Transparency requirements make a black-box decision process legally or reputationally unacceptable |

### 2.2 Architecture trade-offs to weigh before committing scope

| Choice | Option A | Option B | What it trades off |
|---|---|---|---|
| Task scope | Task-specific agent | General-purpose agent | Specialization/reliability vs flexibility/reuse |
| Decision logic | Rule-based | Learning-based | Predictability/auditability vs adaptability |
| Human role | Full autonomy | Human intervention required | Speed vs safety/control |
| System boundary | Open architecture (can call new tools dynamically) | Closed architecture (fixed, pre-approved tool set) | Capability growth vs security/predictability |

A PM's job here is not to pick the "best" option in each row — it's to pick the point on each spectrum that matches the actual risk tolerance of the use case, and to make sure engineering builds to that explicit choice rather than defaulting to "as autonomous as possible."

### 2.3 Build vs buy decision framework

| Question | Lean Build | Lean Buy |
|---|---|---|
| Does this workflow touch proprietary data or process unique to us? | Yes | No |
| Is there already a mature vendor/marketplace agent for this exact task? | No | Yes |
| Do we need deep customization of reasoning/tool logic? | Yes | No |
| What's our tolerance for ongoing maintenance overhead (Chapter 5)? | High | Low |
| How fast do we need this live? | Can wait | Need it now |

A pragmatic middle path used often in practice: buy/customize a pre-built agent framework or platform for the orchestration plumbing, and build only the business-specific tools and prompts on top of it — this mirrors the "framework vs from-scratch" decision engineering already makes technically.

### 2.4 Finding agent opportunities inside your org

1. **Workflow audit** — list workflows that are manual today, cross multiple systems, and have measurable volume.
2. **Filter by repetition + variation** — pure repetition goes to simple automation (2.1); pure novelty is not agent-ready yet; the overlap zone is the agent opportunity.
3. **Check data readiness** — does the workflow have accessible, reasonably clean data/APIs to act on (this is a direct dependency on the RAG and tool-integration work covered in the engineering manual)?
4. **Estimate volume and cost per instance** — this becomes the ROI baseline in Chapter 3.

## Chapter 3: Planning & Delivering an Agent Program

### 3.1 Human-to-agent delegation

Not every task should be fully delegated. Frame delegation as a spectrum, not a switch:

| Level | Description | Example |
|---|---|---|
| Assistive | Agent drafts/suggests, human approves every action | Agent drafts a refund email, human sends it |
| Supervised autonomy | Agent acts within pre-approved bounds, human reviews on exceptions only | Agent processes refunds under $50 automatically, escalates above that |
| Full autonomy | Agent acts without per-instance human review | Agent auto-resolves password reset requests |

This directly maps to the human-in-the-loop gates covered technically in the systems design manual — as the PM, your job is to decide where on this spectrum a given workflow should sit, and that decision becomes an explicit requirement, not an engineering afterthought.

### 3.2 Metrics that actually matter for an agent program

| Category | Metric | Why it matters |
|---|---|---|
| Business impact | Cost saved per resolved task, time saved per task | The core ROI number leadership will ask for |
| Adoption | Usage frequency, % of eligible tasks routed to the agent | An agent nobody uses delivers zero ROI regardless of quality |
| Quality | Task success rate, average resolution duration | Is it actually doing the job well, not just doing it |
| Trust | User satisfaction score, escalation/override rate | High override rate signals the agent isn't trusted yet — often the earliest warning sign of a failing rollout |

### 3.3 Estimating ROI honestly

| Workflow type | ROI likelihood |
|---|---|
| High-volume, repetitive transactions (password resets, status checks) | High — clear time saved, low risk per instance |
| Low-transaction-volume but high manual effort per instance (contract review) | Moderate — savings are real but harder to measure at scale |
| Rapidly changing processes (frequent policy/process changes) | Lower and slower — agent and its prompts need continuous updates, eating into savings |
| Extreme accuracy requirement, low volume | Often negative ROI — the review/oversight cost can exceed the automation benefit |

Two under-counted cost lines PMs frequently miss when building the ROI case: ongoing prompt/model maintenance (Chapter 5.4) and the human review capacity still required at whatever autonomy level was chosen (3.1).

### 3.4 A pragmatic pilot-to-scale playbook

1. **Discover** — pick one workflow from the audit (2.4) with clear volume and measurable baseline cost.
2. **Build with a defined purpose** — set explicit scope, guardrails, and the autonomy level (3.1) before development starts, not after.
3. **Launch narrow** — pilot with a small user group or transaction subset first.
4. **Review feedback and monitor** — track the metrics in 3.2 weekly during the pilot, not just at the end.
5. **Plan ahead** — decide the next workflow to expand to before declaring the pilot "done," so momentum doesn't stall.
6. **Celebrate and communicate** — visible wins build the internal trust needed for the next, larger rollout.

### 3.5 Team shape around an agent program

| Role/function | What it owns |
|---|---|
| Orchestrator (system-level) | The logic that routes tasks to the right specialist agent — a technical component, but its design (who decomposes goals, how agents are selected) is a product decision |
| Specialist agents | Narrow, well-defined capabilities (a research agent, a data-lookup agent) — treat each as its own small product with its own scope and owner |
| Human reviewers | Staff handling escalations and exceptions per the autonomy level set in 3.1 — plan their capacity explicitly, don't assume it shrinks to zero |
| Engineering | Owns implementation quality — guardrails, monitoring, reliability (both covered in the engineering manuals) |
| Product/Project Manager | Owns scope, autonomy-level decisions, ROI tracking, and translating between business goals and technical constraints |

## Chapter 4: Governance, Risk & Compliance

### 4.1 Dimensions of AI governance to define upfront

| Dimension | Question to answer before launch |
|---|---|
| Human involvement | Is a human in the loop, out of the loop, or only pulled in above a defined threshold? |
| Urgency/reversibility | If the agent acts wrongly, how quickly and cheaply can it be corrected? |
| Right to be forgotten | Can a user request their data (including anything the agent stored in memory) be deleted, and does the system support that deletion? |

### 4.2 Regulatory landscape overview (know these exist, don't need the legal detail)

| Regulation/policy | Core idea | Why it matters to a PM |
|---|---|---|
| EU AI Act | Risk-tiered regulation of AI systems, stricter obligations for higher-risk use cases | Determines what documentation/transparency obligations apply based on your use case's risk tier |
| California AI Transparency Act | Requires disclosure when a user is interacting with AI-generated content/systems | Affects UX copy and disclosure requirements for any customer-facing agent |
| China's AI content labelling rules | Requires labelling of AI-generated content | Relevant if the product operates in or serves that market |

Treat this table as a signal to loop in legal/compliance early for any customer-facing or high-risk agent, not as a substitute for actual legal review.

### 4.3 A general governance framework for scoring any agent initiative

| Factor | What to assess |
|---|---|
| Transparency | Can you explain, to a user or auditor, why the agent took a given action? |
| Accountability | Is it clear who (which team, which human reviewer) owns the outcome of an agent's action? |
| Autonomy/risk tier | How much unsupervised action is this agent allowed, relative to how reversible/costly a mistake would be? |

Score every new agent initiative against these three factors before approving its autonomy level (3.1) — a low-transparency, low-accountability, high-autonomy combination is the highest-risk profile and should require the most scrutiny before launch.

### 4.4 What "monitoring" should mean to a PM (translating engineering monitoring into oversight questions)

| Monitoring category | The PM-level question to ask engineering |
|---|---|
| Operational/performance monitoring | Are we tracking latency, cost, and error rate, and do we get alerted before users notice a problem? |
| Ethical and safety monitoring | Is there a check for harmful, biased, or policy-violating output before it reaches a user? |
| State and belief monitoring | Does the agent's internal understanding of a task's state get logged, so we can audit why it made a decision? |
| Anomaly and explainability monitoring | If the agent behaves unexpectedly, can we trace back to the specific step that caused it? |
| Interaction and coordination monitoring (multi-agent only) | If multiple agents are involved, can we see the handoffs between them, not just the final output? |

This table is your translation layer — the engineering manuals implement each of these categories technically; your job is to make sure each one has an owner and a defined response process, not just a dashboard.

## Chapter 5: Risks & Challenges to Manage

### 5.1 Hallucination

What it is in plain terms: the agent states something false with full confidence. The engineering mitigation is retrieval-grounded answers with citations and validation loops (covered in depth in the engineering manual's RAG chapter) — as a PM, your job is to require that any customer-facing factual claim be traceable to a source, and to treat "we added RAG" as a mitigation, not a guarantee.

### 5.2 Security and data exposure

Agents that call tools and APIs are a new attack surface, not just a new feature. Ask specifically: what credentials does this agent hold, what's the blast radius if it's tricked into misusing a tool, and is every irreversible action gated by human approval (ties directly to the guardrails chapter in the systems design manual).

### 5.3 Data quality and bias

An agent trained or grounded on biased or incomplete data will produce biased or incomplete outputs, confidently. Before launch, ask whether the data/documents the agent retrieves from are current, representative, and free of the kind of gaps that would silently skew answers for certain user groups.

### 5.4 Complexity, model drift, and ongoing maintenance

Simple framing: shipping the agent is not the finish line, it's the start of an ongoing maintenance relationship — the same way shipping a mobile app doesn't mean you stop maintaining it. Underlying models change behavior over time (model drift), source data goes stale, and prompts need revision as edge cases surface. Budget a recurring maintenance capacity, not just a one-time build cost.

### 5.5 Trust, transparency, and psychological impact

Employees and customers can react to agent-driven decisions with suspicion if they don't understand how or why a decision was made. Two practical mitigations: communicate clearly what the agent does and doesn't decide, and keep a visible, easy escalation path to a human — trust is built by consistent, explainable behavior over time, not by a single announcement.

## Chapter 6: Leading Teams Through Agentic Transformation

### 6.1 Reframe the agent as a team member, not just a tool

Practically this means: give it a defined scope of responsibility, a clear escalation path, and a review process — the same structure you'd give a new hire, not the structure you'd give a new software license.

### 6.2 Common organizational challenges

| Challenge | What it looks like |
|---|---|
| People and change resistance | Staff worry about being replaced rather than augmented, and disengage from adoption |
| Trust deficit | Teams quietly route around the agent ("shadow" manual processes) instead of using it, undermining the ROI case |
| Multi-agent visibility gap | As more agents get deployed informally across teams, nobody has a full picture of what's running where ("shadow AI") |
| Productivity measurement gap | Leadership expects immediate productivity gains without accounting for the ramp-up and trust-building period |

### 6.3 A change management approach that actually works

1. State plainly what the agent will and won't do, and what happens when it's uncertain.
2. Involve the team whose workflow is being augmented in defining the agent's scope, not just in using the finished product.
3. Position it explicitly as complementing the team's skill, not replacing their judgment, and follow through on that in how autonomy levels are set (3.1).
4. Keep a visible feedback channel and actually act on the feedback — this is what converts skepticism into adoption over a few review cycles.

### 6.4 What to ask engineering before greenlighting a project (bridge to the technical manuals)

| Question | Where the answer lives technically |
|---|---|
| How does the agent remember context across a conversation or task? | Memory systems chapter, engineering manual |
| What happens if a tool call fails mid-task? | Reliability engineering chapter, systems design manual |
| What stops the agent from taking an irreversible action without approval? | Guardrails chapter, systems design manual |
| How will we know if quality degrades after a prompt or model change? | CI/CD and evaluation chapters, systems design manual |
| What's the actual per-request cost, and is there a budget cap? | Cost architecture chapter, systems design manual |

Asking these five questions before approving scope turns a vague "let's add AI to this" request into a scoped, reviewable engineering plan.

## Chapter 7: Practical Decision Toolkit

### 7.1 Quick checklist — should this become an agent project

- The task is repetitive but not so fixed that a simple script already handles it.
- The task spans multiple systems or requires judgment a rule-based automation can't handle.
- Volume/cost data exists to build a real ROI case, not just intuition.
- A clear autonomy level (assistive, supervised, full) can be assigned honestly given the risk.
- Legal/compliance has been looped in if the use case is customer-facing or high-risk.

### 7.2 Quick checklist — build vs buy

- Is there a mature existing agent/vendor for this exact task.
- Does this workflow touch proprietary data or logic that differentiates us.
- Do we have the ongoing maintenance capacity a custom build requires.
- What's the actual time-to-value difference between the two paths.

### 7.3 Quick checklist — before launch

- Guardrails are in place for both input and output, and irreversible actions are gated.
- Monitoring covers all five categories from Chapter 4.4, each with an owner.
- Success metrics (3.2) are defined and instrumented before launch, not after.
- A rollback plan exists if the agent underperforms post-launch.
- The team affected by the workflow change has been part of the rollout, not just informed of it.

### 7.4 Minimal KPI dashboard template for an agent program

| Metric | Target | Actual | Owner |
|---|---|---|---|
| Task success rate | | | |
| Average resolution time | | | |
| Cost per resolved task | | | |
| Adoption rate (% eligible tasks routed to agent) | | | |
| Escalation/override rate | | | |
| User satisfaction score | | | |

### 7.5 Glossary for cross-functional conversations

| Term | Plain meaning |
|---|---|
| Context window | How much conversation/document text the model can consider at once |
| Tool/function calling | The mechanism letting the agent request an action (like a database lookup) instead of just generating text |
| RAG (Retrieval Augmented Generation) | Grounding the agent's answers in your actual documents instead of relying on what the model memorized during training |
| Orchestration | The logic deciding which agent or step handles a given task next |
| Guardrails | Checks that filter or block unsafe/incorrect input and output |
| Human-in-the-loop | A required human approval step before an action is taken |
| Model drift | A model's behavior gradually changing/degrading over time, requiring re-evaluation |
| Hallucination | The model confidently stating something false or unsupported |
| Checkpointing/thread | The mechanism that lets an agent's conversation or task state persist and resume |
