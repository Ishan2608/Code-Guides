# Agentic AI — LangChain, LangGraph & MCP Engineering Manual

## Chapter 1: Foundations & Terminology

### 1.1 AI, ML, DL, GenAI, Agentic AI

Simple explanation: think of these as nested circles, each one a more specific version of the one before it — like "vehicle" containing "car" containing "electric car."

Technical definitions:

| Term | Definition | Key Trait |
|---|---|---|
| AI | Any system performing tasks that normally require human intelligence | Umbrella term |
| ML | Subset of AI; learns patterns from data instead of hardcoded rules | Learns from data |
| DL | Subset of ML using multi-layer neural networks | Learns hierarchical features |
| GenAI | Subset of DL; models that generate new content instead of just classifying/predicting | Generative output |
| Agentic AI | A system built around a GenAI model that can reason, plan, use tools, and act autonomously across multiple steps toward a goal | Autonomy + tool use + loop |

Layered view:

```
AI
 └── ML
      └── DL
           └── GenAI (LLMs, Diffusion models)
                └── Agentic AI (LLM + memory + tools + planning loop)
```

### 1.2 What actually makes something an "AI Agent"

Simple explanation: a single question-answer with an LLM is like asking someone a trivia question. An agent is like handing someone a task and letting them figure out the steps, use a calculator or phone a friend, and check their own work before reporting back.

Technical definition: an Agent = LLM (reasoning engine) + Tools (capabilities) + Memory (state) + a control loop that repeats **Perceive → Reason → Act → Observe** until a goal is satisfied or a stop condition is hit. This loop, known as ReAct, is the mechanism underneath nearly every agent framework, including the ones built later in this manual.

### 1.3 Roles in the field

| Role | Focus |
|---|---|
| ML Engineer | Trains/fine-tunes models, data pipelines, metrics |
| Prompt Engineer | Crafts/optimizes prompts, few-shot design, output evaluation |
| AI/LLM Engineer | Builds applications on LLMs — chains, RAG, agents |
| Agent/System Architect | Designs multi-agent systems, orchestration graphs, state machines |
| LLMOps/MLOps Engineer | Deployment, monitoring, tracing, cost/latency optimization |

### 1.4 Where each tool in this manual fits

| Tool | Role in stack |
|---|---|
| LangChain | Building blocks: model wrappers, prompts, memory, chains, RAG, tool-calling agents |
| LangGraph | Orchestration layer that turns those building blocks into stateful graphs with branches, loops, persistence, and human-in-the-loop control |
| MCP | A standard protocol so any agent framework can plug into any external tool or data source without writing custom integration code for each pair |

Stack visualization:

```
        Your Application (API layer)
                    |
        LangGraph  (orchestration: control flow, state, loops)
                    |
        LangChain  (primitives: models, prompts, memory, tools, RAG)
                    |
        MCP Servers (standardized external capabilities)
                    |
        LLM Provider (OpenAI, Anthropic, etc.)
```

## Chapter 2: The LLM Interfacing Layer

### 2.1 Simple chat model call

Simple explanation: this is like calling a function that takes text in and gives text back — the "function" just happens to be a hosted language model.

General syntax:

```python
from langchain_<provider> import Chat<Provider>

model = Chat<Provider>(model="<model-name>", temperature=0.7)
response = model.invoke("your prompt")
print(response.content)
```

Real example:

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
response = model.invoke("Explain CAP theorem in 2 lines")
print(response.content)
```

Key points:
- `.invoke()` is synchronous single-call. `.stream()` yields chunks. `.batch()` runs many inputs concurrently.
- Every provider integration (`langchain_openai`, `langchain_anthropic`, `langchain_google_genai`, `langchain_groq`) implements the same `BaseChatModel` interface. Swapping providers only means swapping one import and one class name — this uniformity is the core value proposition of LangChain's model layer.

### 2.2 Custom LLM Provider class

You reach for this when your company runs an in-house model server, or a provider has no official LangChain integration.

Simple explanation: you are writing an adapter — same idea as writing a wrapper function around a raw `fetch()` call to an internal API so the rest of your app can call it the same way it calls any other API client.

General syntax:

```python
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

class MyCustomChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = call_my_internal_api(messages)
        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self):
        return "my-custom-model"
```

Real example (wrapping an internal HTTP inference server):

```python
import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

class InternalLlamaModel(BaseChatModel):
    endpoint: str = "http://internal-inference:8000/generate"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        prompt = "\n".join(f"{m.type}: {m.content}" for m in messages)
        resp = requests.post(self.endpoint, json={"prompt": prompt}).json()
        message = AIMessage(content=resp["text"])
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self):
        return "internal-llama"

model = InternalLlamaModel()
model.invoke("Summarize this ticket")
```

Real companies rarely call OpenAI directly everywhere — self-hosted or fine-tuned models get wrapped behind this same interface so prompts, chains, and agents built on top never need to know which provider is underneath.

### 2.3 Streaming and batching

```python
for chunk in model.stream("Write a haiku"):
    print(chunk.content, end="", flush=True)

results = model.batch(["prompt 1", "prompt 2", "prompt 3"])
```

## Chapter 3: Prompt & Message Engineering

### 3.1 PromptTemplate vs ChatPromptTemplate

| Type | Use case |
|---|---|
| `PromptTemplate` | Single string prompt, for legacy completion-style LLMs |
| `ChatPromptTemplate` | List of role-based messages, for chat models — the standard for real work |

General syntax:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "<system instructions with {variables}>"),
    ("human", "{user_input}"),
])
formatted = prompt.invoke({"variables": "...", "user_input": "..."})
```

Real example:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a support agent for {company}. Be concise, cite ticket IDs when relevant."),
    ("human", "{query}"),
])

chain = prompt | model
chain.invoke({"company": "Acme Cloud", "query": "My deployment failed with error 502"})
```

### 3.2 Message types

Simple explanation: think of a conversation as a chat log with labeled speakers — the labels are what these classes represent.

| Message | Meaning |
|---|---|
| `SystemMessage` | Instructions/persona, set once |
| `HumanMessage` | User turn |
| `AIMessage` | Model's turn (may contain `tool_calls`) |
| `ToolMessage` | Result of a tool execution, linked to a `tool_call_id` |

This four-message model is what every memory array, agent loop, and graph state stores internally — everything downstream is just "a growing list of these objects."

### 3.3 Few-shot prompting

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "I want a refund", "output": "intent: refund_request"},
    {"input": "App keeps crashing", "output": "intent: bug_report"},
]
example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
few_shot = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the intent."),
    few_shot,
    ("human", "{input}"),
])
```

### 3.4 Structured output

Simple explanation: instead of asking the model to write a paragraph and hoping you can regex out the useful part, you ask it to fill in a form (a schema) — you get back a Python object, not raw text.

General syntax:

```python
from pydantic import BaseModel

class MySchema(BaseModel):
    field: str

structured_model = model.with_structured_output(MySchema)
result = structured_model.invoke("...")   # returns a MySchema instance, not text
```

Real example:

```python
from pydantic import BaseModel, Field

class TicketClassification(BaseModel):
    intent: str = Field(description="one of: refund_request, bug_report, feature_request")
    urgency: int = Field(description="1 (low) to 5 (critical)")

classifier = model.with_structured_output(TicketClassification)
result = classifier.invoke("The app crashed and I lost all my saved work, this is urgent!")
print(result.intent, result.urgency)
```

`with_structured_output` is the pattern used constantly for routing decisions inside chains and agents — it removes the need to parse free text at all.

### 3.5 Output parsers (when you are not using tool-call-based structured output)

| Parser | Purpose |
|---|---|
| `StrOutputParser` | Extracts plain `.content` string from an `AIMessage`, used to end most simple chains |
| `JsonOutputParser` | Parses model output as JSON, optionally validated against a Pydantic schema |
| `RetryOutputParser` / `OutputFixingParser` | Re-prompts the model automatically if its output fails to parse — a safety net for less reliable models |

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser(pydantic_object=TicketClassification)
chain = prompt | model | parser
```

## Chapter 4: Memory Systems

### 4.1 Simulating memory with a plain array

Simple explanation: an LLM has no memory between calls, the same way a stateless API endpoint forgets everything after responding. "Memory" is really just you resending the whole conversation transcript every time.

```python
history = []

def chat(user_input):
    history.append(HumanMessage(content=user_input))
    response = model.invoke(history)
    history.append(response)
    return response.content
```

Every memory abstraction below does exactly this internally — a growing list of messages re-sent on every call.

### 4.2 `RunnableWithMessageHistory`

Older `ConversationBufferMemory`-style classes are deprecated. The current pattern wraps any chain with history management keyed by a session id.

General syntax:

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

chain_with_history.invoke(
    {"input": "..."},
    config={"configurable": {"session_id": "user_123"}},
)
```

### 4.3 Per-user session isolation

The `session_id` is what guarantees User A never sees User B's chat — this is the actual mechanism a multi-tenant chatbot backend relies on.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
base_chain = prompt | model

chatbot = RunnableWithMessageHistory(
    base_chain, get_session_history,
    input_messages_key="input", history_messages_key="history",
)

chatbot.invoke({"input": "My name is Riya"}, config={"configurable": {"session_id": "user_101"}})
chatbot.invoke({"input": "What's my name?"}, config={"configurable": {"session_id": "user_202"}})
# user_202 gets "I don't know your name" -- correct isolation
```

### 4.4 Persistent memory backends

`InMemoryChatMessageHistory` dies on server restart. Swap the backing store, keep everything else identical:

```python
from langchain_mongodb import MongoDBChatMessageHistory

def get_session_history(session_id: str):
    return MongoDBChatMessageHistory(
        connection_string="mongodb://localhost:27017",
        session_id=session_id,
        database_name="chatapp",
        collection_name="chat_histories",
    )
```

Mental model: `session_id` is a primary key. In a real app it maps 1:1 to `user_id` or `conversation_id` from your auth system — the LLM layer never invents identity, it only reads what the application layer passes in.

### 4.5 Trimming and windowing

```python
from langchain_core.messages import trim_messages

trimmer = trim_messages(max_tokens=2000, strategy="last", token_counter=model)
trimmed_history = trimmer.invoke(full_history)
```

### 4.6 Summarization memory (for conversations too long to trim safely)

Simple explanation: instead of dropping old messages entirely, you compress them into a running summary — like keeping meeting minutes instead of a full transcript.

```python
def summarize_if_needed(history: list, model) -> list:
    if len(history) < 20:
        return history
    old, recent = history[:-6], history[-6:]
    summary = model.invoke(f"Summarize this conversation briefly:\n{old}")
    return [SystemMessage(f"Summary of earlier conversation: {summary.content}")] + recent
```

This is called before each `.invoke()` in a long-running session to keep token cost and latency bounded while preserving relevant earlier context.

## Chapter 5: Chains — Composition Patterns (LCEL)

Simple explanation: a chain is just functions piped together, the same way you'd pipe shell commands (`cmd1 | cmd2 | cmd3`) or chain `.then()` calls on a JavaScript Promise.

Technical model: everything in LangChain is a `Runnable`. `A | B` means `B.invoke(A.invoke(input))`.

### 5.1 Sequential chain

General syntax:

```python
chain = step1 | step2 | step3
chain.invoke(input)
```

Real example:

```python
extract_topic = ChatPromptTemplate.from_template("Extract the main topic from: {text}") | model
write_summary = ChatPromptTemplate.from_template("Write a 1-line summary about: {topic}") | model

full_chain = (
    {"topic": extract_topic | (lambda x: x.content)}
    | write_summary
)
full_chain.invoke({"text": "Long customer complaint about billing..."})
```

### 5.2 Parallel chain (`RunnableParallel`)

Simple explanation: like `Promise.all()` — run independent branches at the same time instead of one after another.

General syntax:

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(branch_a=chain_a, branch_b=chain_b)
result = parallel.invoke(input)   # {"branch_a": ..., "branch_b": ...}
```

Real example (analyze sentiment and category of a ticket simultaneously — halves latency versus doing it sequentially):

```python
sentiment_chain = ChatPromptTemplate.from_template("Sentiment of: {text}") | model
category_chain = ChatPromptTemplate.from_template("Category of: {text}") | model

analysis = RunnableParallel(sentiment=sentiment_chain, category=category_chain)
result = analysis.invoke({"text": "This is the third time your app has crashed on me!"})
```

### 5.3 Branching (`RunnableBranch`)

General syntax:

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (condition_fn_1, chain_1),
    (condition_fn_2, chain_2),
    default_chain,
)
```

Real example (route by ticket urgency computed via structured output):

```python
branch = RunnableBranch(
    (lambda x: x["urgency"] >= 4, escalation_chain),
    (lambda x: x["intent"] == "refund_request", refund_chain),
    general_response_chain,
)
```

### 5.4 Loops, and where plain chains stop being enough

You can hack a loop with a Python `while` around `.invoke()`:

```python
result = chain.invoke(input)
while not is_good_enough(result):
    result = chain.invoke(refine_input(result))
```

This has no persisted state between steps, no resumability if the process crashes mid-loop, no visibility into intermediate steps, and no way to pause for a human to approve a step. Those four limitations are the actual reasons a dedicated orchestration layer with explicit state and control flow becomes necessary once an application needs loops, branching multi-step reasoning, or multiple cooperating agents — this is what the LangGraph chapters build properly.

## Chapter 6: RAG — Retrieval Augmented Generation

Simple explanation: RAG is an open-book exam instead of a closed-book one. The model doesn't need to have memorized your company's documents — it just gets handed the relevant page right before answering.

Technical flow:

```
User Question
     |
     v
Embed question --> Vector search over indexed docs --> Top-k relevant chunks
     |
     v
Inject chunks into prompt as context --> LLM generates grounded answer
```

RAG solves: LLMs don't know your private or recent data. Instead of fine-tuning (expensive, static), you retrieve relevant text at query time and place it into the prompt.

### 6.1 Minimal RAG pipeline

General syntax:

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

docs = TextLoader("file.txt").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

relevant_docs = retriever.invoke("user question")
```

Real example — end-to-end minimal RAG chain:

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY this context:\n{context}\n\nQuestion: {question}"
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)
rag_chain.invoke("What is our refund policy for annual plans?")
```

### 6.2 Document loading and splitting

| Decision | Options | Rule of thumb |
|---|---|---|
| Loader | `PyPDFLoader`, `TextLoader`, `WebBaseLoader`, `Docx2txtLoader` | Match to source format |
| Chunk size | 300-1500 chars | Smaller = precise retrieval, weaker context. Larger = opposite |
| Overlap | 10-20% of chunk size | Prevents cutting a fact/sentence across a chunk boundary |
| Splitter | `RecursiveCharacterTextSplitter` (general), `MarkdownHeaderTextSplitter` (structured docs) | Use structure-aware splitters when the source has headers/sections |

### 6.3 Vector stores for real deployments

| Store | When to use |
|---|---|
| Chroma / FAISS | Local dev, prototypes, small corpora |
| MongoDB Atlas Vector Search | You already store app data in MongoDB — keep everything in one database, add a `$vectorSearch` index |
| Pinecone / Weaviate / Qdrant | Managed, large-scale production vector search |

Real example using MongoDB:

```python
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
collection = client["chatapp"]["knowledge_base"]

vectorstore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=OpenAIEmbeddings(),
    index_name="vector_index",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

### 6.4 Integrating RAG into a session-aware chatbot

Goal: the same per-user chatbot now also answers from documents. Pattern: retrieve context on every turn and inject it alongside chat history.

```python
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a support assistant. Use the provided context if relevant, "
               "otherwise answer normally.\n\nContext:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

def build_input(payload):
    docs = retriever.invoke(payload["input"])
    payload["context"] = format_docs(docs)
    return payload

rag_chatbot_chain = build_input | rag_prompt | model

rag_chatbot = RunnableWithMessageHistory(
    rag_chatbot_chain, get_session_history,
    input_messages_key="input", history_messages_key="history",
)

rag_chatbot.invoke(
    {"input": "What's the refund window for annual plans?"},
    config={"configurable": {"session_id": "user_101"}},
)
rag_chatbot.invoke(
    {"input": "And what if I already used it for 2 months?"},
    config={"configurable": {"session_id": "user_101"}},
)
```

Retrieval is simply another input transform bolted onto an existing session-aware chain — the memory mechanism underneath is unchanged.

### 6.5 Beyond naive RAG

| Technique | Fixes |
|---|---|
| Reranking (e.g. Cohere rerank, cross-encoder) | Vector similarity is not the same as true relevance; rerank top-k before feeding to the LLM |
| Hybrid search (vector + keyword/BM25) | Pure embeddings miss exact term or ID matches |
| Self-query retriever | Extracts structured filters (dates, categories) from natural language before running vector search |
| Parent-document retriever | Retrieves small precise chunks but returns larger parent context to the LLM |

### 6.6 Evaluating RAG quality

Simple explanation: you need a way to tell if the retriever is even fetching the right pages before blaming the LLM for a bad answer.

| Metric | What it checks |
|---|---|
| Retrieval precision/recall | Did the retrieved chunks actually contain the answer |
| Faithfulness | Does the generated answer stay grounded in the retrieved context, or does it hallucinate beyond it |
| Answer relevance | Does the answer actually address the question asked |

Frameworks such as RAGAS or LangSmith's evaluation datasets automate scoring these on a labeled question/answer set — worth setting up before trusting a RAG pipeline in production.

## Chapter 7: Tools & AI Agents

Simple explanation: up to now the model only produces text. A tool is like giving it access to a calculator, a phone, or a database query — the model decides when it needs one, your code actually runs it, and hands the result back.

Technical flow:

```
User -> LLM (sees available tools) -> decides: "call tool X with args Y"
     -> your code executes tool X -> result returned to LLM as ToolMessage
     -> LLM produces final answer (or calls another tool)
```

### 7.1 Defining tools

General syntax:

```python
from langchain_core.tools import tool

@tool
def tool_name(arg: type) -> return_type:
    """Docstring -- the LLM reads this to decide when to use the tool."""
    ...
    return result
```

Real example:

```python
@tool
def check_order_status(order_id: str) -> str:
    """Look up the current status of an order by its ID."""
    order = db.orders.find_one({"_id": order_id})
    return order["status"] if order else "Order not found"

@tool
def calculate_refund(order_total: float, days_used: int) -> float:
    """Calculate refund amount based on pro-rated usage."""
    return round(order_total * max(0, (30 - days_used) / 30), 2)
```

### 7.2 The manual tool-calling loop

Understanding this once demystifies every "agent" abstraction that follows.

```python
tools = [check_order_status, calculate_refund]
model_with_tools = model.bind_tools(tools)

messages = [HumanMessage("What's the status of order A123?")]
response = model_with_tools.invoke(messages)
messages.append(response)

for call in response.tool_calls:
    tool_fn = {"check_order_status": check_order_status,
               "calculate_refund": calculate_refund}[call["name"]]
    result = tool_fn.invoke(call["args"])
    messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

final = model_with_tools.invoke(messages)
```

### 7.3 Prebuilt ReAct agent

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools=[check_order_status, calculate_refund])
agent.invoke({"messages": [HumanMessage("Order A123 was used for 10 days, what refund do they get?")]})
```

`create_react_agent` lives in `langgraph`, not classic LangChain `AgentExecutor` — this is the current, actively maintained path.

### 7.4 Upgrading a chatbot into a tool-using agent

```python
@tool
def search_knowledge_base(query: str) -> str:
    """Search company docs for policy/product information."""
    docs = retriever.invoke(query)
    return format_docs(docs)

tools = [search_knowledge_base, check_order_status, calculate_refund]
support_agent = create_react_agent(model, tools=tools)

support_agent.invoke({
    "messages": [
        SystemMessage("You are Acme's support agent. Use tools when needed."),
        HumanMessage("My order A123 -- what's the status, and what's the refund policy?"),
    ]
})
```

Retrieval becomes just another tool (`search_knowledge_base`) instead of a hardcoded prompt injection, chosen by the LLM itself rather than always running on every turn. Session-scoped memory still applies here the same way — wrap this agent with `RunnableWithMessageHistory` for per-user isolation in a real deployment.

### 7.5 Tool error handling

Simple explanation: a tool call is an external call, and external calls fail — treat it like any API call in a normal backend, with try/except, not like a guaranteed function return.

```python
@tool
def check_order_status(order_id: str) -> str:
    """Look up the current status of an order by its ID."""
    try:
        order = db.orders.find_one({"_id": order_id})
        return order["status"] if order else f"No order found with id {order_id}"
    except Exception as e:
        return f"Tool error: could not fetch order status ({e})"
```

Returning a descriptive string instead of raising lets the agent see the failure and decide how to recover (retry, ask the user for clarification, or apologize) instead of crashing the whole request.

### 7.6 Structured tool arguments

For tools with multiple or complex arguments, define an explicit Pydantic schema instead of relying on type hints alone — this gives the model clearer field descriptions and gives you validation.

```python
from pydantic import BaseModel, Field

class RefundArgs(BaseModel):
    order_total: float = Field(description="Total amount paid for the order")
    days_used: int = Field(description="Number of days the product/service was used")

@tool(args_schema=RefundArgs)
def calculate_refund(order_total: float, days_used: int) -> float:
    """Calculate refund amount based on pro-rated usage."""
    return round(order_total * max(0, (30 - days_used) / 30), 2)
```

## Chapter 8: Multi-Agent Architecture

### 8.1 Why split into multiple agents

Simple explanation: one person trying to be a doctor, lawyer, and accountant at once gives worse answers than three specialists working together with someone routing questions to the right one.

Technical reasons: a single agent with 30 tools has poor tool-selection accuracy and a bloated system prompt. Multi-agent systems assign each sub-agent a narrow toolset and focused prompt, with a router or supervisor deciding who handles what.

### 8.2 Core patterns

| Pattern | Structure | Use case |
|---|---|---|
| Supervisor | One manager agent routes tasks to specialist agents, aggregates results | Most common in production — support triage, coding assistants |
| Hierarchical | Supervisors of supervisors | Large systems with many domains |
| Network | Agents call each other directly, no central router | Research/debate-style agents; harder to control, less common in production |

Visualization — supervisor pattern:

```
                     Supervisor
     User  ------->      |
                  --------+--------
                  |       |        |
              BillingAgt TechAgt RefundAgt
```

### 8.3 Building this with plain Python, and where it strains

```python
router_chain = ChatPromptTemplate.from_template(
    "Route this query to one of: billing, technical, refund.\nQuery: {query}\nAnswer with one word."
) | model | StrOutputParser()

def route(query):
    target = router_chain.invoke({"query": query}).strip().lower()
    agent = {"billing": billing_agent, "technical": tech_agent, "refund": refund_agent}[target]
    return agent.invoke({"messages": [HumanMessage(query)]})
```

This works for a single routing hop, but breaks down once the system needs shared state across agents, an agent handing control back to the supervisor for a second pass, loops between agents, or pausing mid-flow for human approval. Plain `if/else` orchestration turns into unmanageable spaghetti quickly once any of those requirements shows up — which is exactly the gap a graph-based orchestration layer with explicit state and control flow is built to close.

## Chapter 9: LangGraph Fundamentals

Simple explanation: a LangGraph app is like a flowchart you actually run. Each box (node) does some work and updates a shared whiteboard (state); arrows (edges) decide which box runs next, and some arrows have conditions.

Technical model:

```
        [Node A]        [Node B]
START ----->  |--------->   |------> END
             (conditional edge, loop possible)
```

### 9.1 Core building blocks

| Concept | Role |
|---|---|
| `State` | A TypedDict/Pydantic schema — the single shared object passed between nodes |
| `Node` | A function `(state) -> partial_state_update` |
| `Edge` | Fixed transition: always go from node X to node Y |
| `Conditional Edge` | A function inspects state and decides the next node |
| `Checkpointer` | Persists state after every step, enabling resume, replay, and human-in-the-loop |

### 9.2 Minimal StateGraph syntax

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    ...

def node_fn(state: State) -> dict:
    ...
    return {"key": new_value}   # partial update, merged into state

graph = StateGraph(State)
graph.add_node("node_name", node_fn)
graph.add_edge(START, "node_name")
graph.add_edge("node_name", END)
app = graph.compile()

app.invoke({"key": initial_value})
```

### 9.3 How state updates actually merge — reducers

Simple explanation: when two nodes both try to update the same field, LangGraph needs a rule for how to combine them — like deciding whether a new value overwrites the whiteboard entry or gets appended underneath it.

By default, a key is simply overwritten by the latest node's return value. For fields like conversation history, you want appending instead of overwriting — this is what the `add_messages` reducer is for:

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]   # new messages are appended, not overwritten
```

Any custom field can define its own reducer function the same way — `Annotated[list, my_merge_fn]` — whenever "append" or a custom merge is needed instead of "replace."

### 9.4 Real example: a critique-and-retry loop

```python
class DraftState(TypedDict):
    topic: str
    draft: str
    feedback: str
    attempts: int

def write_draft(state: DraftState) -> dict:
    result = model.invoke(f"Write a short marketing blurb about {state['topic']}. "
                           f"Previous feedback: {state.get('feedback', 'none')}")
    return {"draft": result.content, "attempts": state.get("attempts", 0) + 1}

def critique(state: DraftState) -> dict:
    result = model.invoke(f"Critique this blurb, say GOOD or give feedback:\n{state['draft']}")
    return {"feedback": result.content}

def should_continue(state: DraftState) -> str:
    if "GOOD" in state["feedback"] or state["attempts"] >= 3:
        return END
    return "write_draft"

graph = StateGraph(DraftState)
graph.add_node("write_draft", write_draft)
graph.add_node("critique", critique)
graph.add_edge(START, "write_draft")
graph.add_edge("write_draft", "critique")
graph.add_conditional_edges("critique", should_continue, {"write_draft": "write_draft", END: END})

app = graph.compile()
result = app.invoke({"topic": "a new noise-cancelling headphone", "attempts": 0})
```

Every step here is a visible, inspectable node with state persisted at each hop, instead of an opaque `while` loop.

## Chapter 10: LangGraph Control Flow & State Design

### 10.1 Conditional edges in depth

A conditional edge function receives the current state and returns the name of the next node (or `END`). It can inspect any field, not just a single flag:

```python
def route_by_confidence(state: State) -> str:
    if state["confidence"] > 0.8:
        return "finalize"
    elif state["retries"] < 2:
        return "gather_more_info"
    else:
        return "escalate_to_human"

graph.add_conditional_edges("assess", route_by_confidence)
```

### 10.2 Recursion limits

Simple explanation: a mis-wired conditional edge can loop forever, the same way a bug in a recursive function without a base case blows the stack — `recursion_limit` is the safety cap.

```python
app.invoke(initial_state, config={"recursion_limit": 25})
```

### 10.3 Subgraphs — nesting graphs as nodes

Simple explanation: a subgraph is a reusable flowchart-within-a-flowchart, the same way you'd extract a block of steps into its own function and call that function from a bigger one.

```python
def build_billing_subgraph():
    sub = StateGraph(BillingState)
    sub.add_node("lookup", lookup_invoice)
    sub.add_node("calculate", calculate_charge)
    sub.add_edge(START, "lookup")
    sub.add_edge("lookup", "calculate")
    sub.add_edge("calculate", END)
    return sub.compile()

billing_subgraph = build_billing_subgraph()

main_graph = StateGraph(MainState)
main_graph.add_node("billing", billing_subgraph)   # a compiled graph can be used directly as a node
```

Subgraphs keep large systems organized — each domain team can own and test its own subgraph independently, and the main graph just wires them together.

### 10.4 Parallel node execution (fan-out / fan-in)

Multiple nodes can branch off the same source node and run concurrently, then converge into one node that reads all their results from state:

```python
graph.add_edge("start", "check_inventory")
graph.add_edge("start", "check_fraud_score")
graph.add_edge("check_inventory", "finalize_order")
graph.add_edge("check_fraud_score", "finalize_order")   # finalize_order waits for both
```

This is the graph-native version of the `RunnableParallel` pattern from the chains chapter, but with each branch able to be an arbitrarily complex subgraph instead of a single chain call.

### 10.5 Combining a state update with routing using `Command`

Simple explanation: instead of returning a plain dict and letting a separate conditional-edge function decide where to go next, a node can just say "here's the new state, and here's exactly where to go next" in one return.

```python
from langgraph.types import Command
from typing import Literal

def assess(state: State) -> Command[Literal["finalize", "escalate_to_human"]]:
    confidence = compute_confidence(state)
    if confidence > 0.8:
        return Command(update={"confidence": confidence}, goto="finalize")
    return Command(update={"confidence": confidence}, goto="escalate_to_human")
```

`Command` is especially useful in multi-agent handoffs, covered in the multi-agent chapter below.

## Chapter 11: LangGraph Persistence & Human-in-the-loop

### 11.1 Checkpointers and threads

Simple explanation: a checkpointer is autosave for the graph's whiteboard — every step gets saved so a conversation (or a long-running task) can be paused and resumed exactly where it left off, even after a server restart.

```python
from langgraph.checkpoint.memory import InMemorySaver
# Production: from langgraph.checkpoint.mongodb import MongoDBSaver

checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user_101"}}
app.invoke({"topic": "..."}, config=config)
app.invoke({"feedback": "make it punchier"}, config=config)  # continues the same thread's state
```

`thread_id` plays the same role here as `session_id` did for chat memory — one identifier, one isolated conversation/task.

### 11.2 Time travel and replay

Because every step is checkpointed, you can inspect or rewind to any prior state — useful for debugging exactly where an agent's reasoning went wrong.

```python
history = list(app.get_state_history(config))   # every checkpoint, most recent first
earlier_state = history[3]
app.invoke(None, config={"configurable": {"thread_id": "user_101", "checkpoint_id": earlier_state.config["configurable"]["checkpoint_id"]}})
```

### 11.3 Pausing for human approval

```python
from langgraph.types import interrupt, Command

def send_refund(state):
    approved = interrupt({"question": f"Approve refund of ${state['refund_amount']}?"})
    if not approved:
        return {"status": "rejected"}
    process_refund(state["order_id"])
    return {"status": "sent"}

app.invoke({"order_id": "A123", "refund_amount": 49.99}, config=config)   # pauses here
app.invoke(Command(resume=True), config=config)   # resumes after human approves
```

This is the standard way production agents avoid autonomously taking irreversible actions — refunds, emails, deployments — without a human checkpoint.

### 11.4 Long-term memory across threads (the `Store` API)

Simple explanation: a checkpointer remembers one conversation thread. A store remembers facts about a user that should carry over into every future conversation, the same way a CRM remembers a customer regardless of which support ticket they're currently on.

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
app = graph.compile(checkpointer=checkpointer, store=store)

def remember_preference(state, config, *, store):
    user_id = config["configurable"]["user_id"]
    store.put(("preferences", user_id), "communication_style", {"style": "concise"})

def use_preference(state, config, *, store):
    user_id = config["configurable"]["user_id"]
    pref = store.get(("preferences", user_id), "communication_style")
    ...
```

Checkpointer state is per-thread; store data is per-user (or any namespace you choose) and persists regardless of which thread is active.

## Chapter 12: LangGraph Multi-Agent Patterns

### 12.1 Supervisor pattern

```python
class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str

def supervisor(state: SupportState) -> dict:
    decision = model.with_structured_output(RouteDecision).invoke(state["messages"])
    return {"next_agent": decision.target}

def billing_node(state): return {"messages": [billing_agent.invoke(state["messages"])]}
def tech_node(state):    return {"messages": [tech_agent.invoke(state["messages"])]}
def refund_node(state):  return {"messages": [refund_agent.invoke(state["messages"])]}

graph = StateGraph(SupportState)
graph.add_node("supervisor", supervisor)
graph.add_node("billing", billing_node)
graph.add_node("technical", tech_node)
graph.add_node("refund", refund_node)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", lambda s: s["next_agent"],
                             {"billing": "billing", "technical": "technical", "refund": "refund"})
graph.add_edge("billing", END)
graph.add_edge("technical", END)
graph.add_edge("refund", END)

support_system = graph.compile(checkpointer=checkpointer)
```

Each specialist node can itself be a full `create_react_agent` with its own tools — graphs compose, so a "node" in one graph can be an entire subgraph or agent.

### 12.2 Hierarchical teams

Simple explanation: the same supervisor idea, but the supervisor's own "workers" are themselves supervisors of smaller teams — like a department head routing to team leads, who then route to individual specialists.

```python
support_team_graph = build_support_team_graph()     # supervisor + billing/tech/refund
engineering_team_graph = build_engineering_team_graph()

top_graph = StateGraph(TopState)
top_graph.add_node("top_supervisor", top_supervisor_node)
top_graph.add_node("support_team", support_team_graph)
top_graph.add_node("engineering_team", engineering_team_graph)
top_graph.add_conditional_edges("top_supervisor", route_to_team,
                                 {"support": "support_team", "engineering": "engineering_team"})
```

### 12.3 Direct handoffs between agents (without a central supervisor)

An agent node can decide to hand control directly to another named agent using `Command(goto=...)`, useful in network-style patterns where control shouldn't always return to a central router:

```python
def billing_agent_node(state) -> Command[Literal["technical", END]]:
    response = billing_agent.invoke(state["messages"])
    if "this looks like a bug, not a billing issue" in response.content.lower():
        return Command(update={"messages": [response]}, goto="technical")
    return Command(update={"messages": [response]}, goto=END)
```

### 12.4 Migrating a chatbot with RAG and tools into one graph

```python
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

agent_node = create_react_agent(model, tools=[search_knowledge_base, check_order_status, calculate_refund])

graph = StateGraph(ChatState)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

app = graph.compile(checkpointer=checkpointer)

app.invoke(
    {"messages": [HumanMessage("What's the refund policy, and check order A123 too")]},
    config={"configurable": {"thread_id": "user_101"}},
)
```

The model, prompt, tools, persistent per-user memory (via `thread_id`), retrieval-as-a-tool, and the agent loop all collapse into this one compiled graph — the artifact that actually gets wrapped in an API and deployed.

## Chapter 13: LangGraph Streaming, Debugging & Testing

### 13.1 Streaming modes

```python
for event in app.stream(
    {"messages": [HumanMessage("...")]},
    config=config,
    stream_mode="values",     # full state after each step
):
    print(event["messages"][-1].content)

for event in app.stream({"messages": [...]}, config=config, stream_mode="messages"):
    # token-by-token streaming of LLM output, useful for a live-typing UI
    ...

for event in app.stream({"messages": [...]}, config=config, stream_mode="updates"):
    # only the incremental change each node produced, useful for logging/tracing
    ...
```

### 13.2 Visualizing a graph

Every compiled graph can render its own diagram, which is worth generating for any non-trivial graph rather than trying to mentally trace conditional edges:

```python
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
```

LangGraph Studio (a visual debugger) additionally lets you step through a running graph node by node and inspect state at each point, similar to a debugger's step-through mode.

### 13.3 Testing nodes in isolation

Simple explanation: test each box in the flowchart the way you would unit test any function — separately from the wiring, before testing the whole graph end to end.

```python
def test_supervisor_routes_billing_query():
    state = {"messages": [HumanMessage("Why was I charged twice?")], "next_agent": ""}
    result = supervisor(state)
    assert result["next_agent"] == "billing"

def test_refund_calculation_tool():
    assert calculate_refund.invoke({"order_total": 100, "days_used": 15}) == 50.0
```

Only after individual nodes and tools are verified does it make sense to test the compiled graph's full `.invoke()` behavior end to end, using a small set of realistic conversation scenarios.

## Chapter 14: MCP — Model Context Protocol

### 14.1 What problem MCP solves

Simple explanation: think "USB-C for AI tool access" — one standard connector instead of a different custom cable for every device.

Technical reasoning: without a shared protocol, every agent framework writes bespoke integration code for every external tool (Slack, GitHub, a filesystem, a database). N frameworks times M tools means N times M integrations. MCP standardizes the interface so any MCP-compatible client can talk to any MCP server, turning that into roughly N plus M.

### 14.2 Architecture

```
   MCP Host                 MCP protocol (JSON-RPC)              MCP Server
 (your agent / IDE)  <-------------------------------->   (exposes tools/resources)
```

| Component | Role |
|---|---|
| Host | The application using the LLM (a LangGraph app, Claude Desktop, an IDE) |
| Client | The library inside the host that speaks MCP to servers |
| Server | Exposes Tools (actions), Resources (readable data), and Prompts (templates) over a standard interface |

### 14.3 Building a minimal MCP server

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("support-tools")

@mcp.tool()
def check_order_status(order_id: str) -> str:
    """Look up order status."""
    order = db.orders.find_one({"_id": order_id})
    return order["status"] if order else "Order not found"

if __name__ == "__main__":
    mcp.run(transport="stdio")   # or "sse" / "streamable-http" for network access
```

The tool logic itself is identical to a native LangChain `@tool` — MCP is a transport and exposure layer, not a new way of thinking about tools.

### 14.4 Consuming MCP tools inside a LangGraph agent

General syntax:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "support": {"command": "python", "args": ["support_server.py"], "transport": "stdio"},
})
mcp_tools = await client.get_tools()
agent = create_react_agent(model, tools=mcp_tools)
```

Real example — combining MCP tools with native tools in one agent:

```python
client = MultiServerMCPClient({
    "support": {"command": "python", "args": ["support_server.py"], "transport": "stdio"},
    "github":  {"url": "https://api.githubmcp.example/mcp", "transport": "streamable_http"},
})
mcp_tools = await client.get_tools()

all_tools = mcp_tools + [search_knowledge_base, calculate_refund]
agent = create_react_agent(model, tools=all_tools)
```

### 14.5 Native tool vs MCP server — decision table

| Situation | Choice |
|---|---|
| Tool is internal, used only by one app | Native `@tool` — simpler, no protocol overhead |
| Tool or data source should be reusable across multiple different agent apps or non-LangChain hosts | MCP server |
| Integrating a third-party service that already ships an MCP server | Use their MCP server directly instead of writing a custom wrapper |

## Chapter 15: Production Engineering

### 15.1 Realistic project structure

```
support-agent/
  app/
    models.py          # chat model setup
    prompts.py         # prompt templates
    memory.py          # get_session_history, checkpointer config
    tools/
      kb_search.py      # RAG-as-tool
      orders.py
      mcp_tools.py      # MCP client setup
    graph.py            # StateGraph definition
    api.py              # FastAPI routes
  ingestion/
    build_index.py      # one-off script: load docs -> chunk -> embed -> vector store
  requirements.txt
```

### 15.2 Session and user management in a real database

| Layer | Storage |
|---|---|
| User accounts / auth | Your existing app database (a `users` collection in MongoDB) |
| Conversation state / checkpoints | `MongoDBSaver` checkpointer, keyed by `thread_id = user_id` (or `conversation_id` if a user has multiple threads) |
| Long-term user facts | LangGraph `Store`, keyed by `user_id`, independent of any single thread |
| Knowledge base (RAG) | MongoDB Atlas Vector Search or a dedicated vector DB, a separate collection from chat state |

### 15.3 Observability

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "..."
os.environ["LANGSMITH_PROJECT"] = "support-agent"
# every .invoke() call is now automatically traced -- no code change needed
```

What to check in a trace: which tool was called and with what arguments, latency per node, and where a chain or graph diverged from expected behavior. This is non-negotiable once an agent has more than a couple of steps or tools — debugging blind is not viable.

### 15.4 Cost, latency, and rate limiting

| Concern | Mitigation |
|---|---|
| An agent can trigger many LLM calls for a single user message | Cap max tool-call iterations per request; log token usage per request |
| Slow perceived response time | Stream tokens to the client (`stream_mode="messages"`) instead of waiting for the full response |
| Cost spikes from retries or loops | Set `recursion_limit`; add per-user rate limiting at the API layer |

### 15.5 Deployment checklist

- Tool errors are caught and returned as a `ToolMessage`, not raised — an agent should see "tool failed: X" and recover, not crash the request.
- `recursion_limit` is set on every graph invocation.
- Rate and cost limits exist per user.
- Any irreversible action (payments, sending external messages, deleting data) goes through `interrupt()` for human approval.
- The RAG index rebuild pipeline is a separate, versioned process from the serving app — never re-embed documents on every request.

## Chapter 16: Capstone — Full System with a Working Backend

### 16.1 End-to-end architecture

```
                    FastAPI (user_id -> thread_id)
                              |
                    LangGraph app (compiled graph)
                              |
                        Supervisor node
                    ----------+----------
                    |         |          |
                [Billing][Technical][Refund]
              each a create_react_agent with its own tools
                    |                    |
         search_knowledge_base    check_order_status
             (RAG tool)             calculate_refund
                    |               (native tools, plus
        Vector store (MongoDB       any MCP-exposed tools)
          Atlas Vector Search)
                              |
              Checkpointer (MongoDBSaver) -- persists per thread_id
```

### 16.2 Complete agent definition

```python
# graph.py
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

class RouteDecision(BaseModel):
    target: Literal["billing", "technical", "refund"] = Field(description="which team should handle this")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str

def build_graph():
    billing_agent = create_react_agent(model, tools=[calculate_refund])
    tech_agent    = create_react_agent(model, tools=[search_knowledge_base])
    refund_agent  = create_react_agent(model, tools=[check_order_status, calculate_refund])

    def supervisor(state: State) -> dict:
        decision = model.with_structured_output(RouteDecision).invoke(state["messages"])
        return {"next_agent": decision.target}

    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor)
    graph.add_node("billing", lambda s: billing_agent.invoke(s))
    graph.add_node("technical", lambda s: tech_agent.invoke(s))
    graph.add_node("refund", lambda s: refund_agent.invoke(s))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", lambda s: s["next_agent"],
        {"billing": "billing", "technical": "technical", "refund": "refund"},
    )
    graph.add_edge("billing", END)
    graph.add_edge("technical", END)
    graph.add_edge("refund", END)

    return graph.compile(checkpointer=InMemorySaver())   # swap for MongoDBSaver in production
```

### 16.3 FastAPI backend exposing the agent to real users

```python
# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from graph import build_graph

app = FastAPI(title="Support Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # restrict to your actual frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

graph_app = build_graph()

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.user_id}, "recursion_limit": 15}
    result = graph_app.invoke({"messages": [HumanMessage(req.message)]}, config=config)
    return ChatResponse(reply=result["messages"][-1].content)

@app.get("/health")
def health():
    return {"status": "ok"}
```

Run it:

```
uvicorn api:app --reload --port 8000
```

### 16.4 Calling it as a client

```
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_101", "message": "What is your refund policy for annual plans?"}'
```

Minimal JavaScript client (what a real frontend would do):

```javascript
async function sendMessage(userId, message) {
  const res = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, message }),
  });
  const data = await res.json();
  return data.reply;
}
```

`user_id` here is whatever your auth system already produces (a session cookie, a JWT claim, a database primary key) — it becomes the `thread_id` that keeps each user's conversation state isolated and persistent across requests. This is the same identity-mapping principle used throughout the memory sections, now sitting behind an actual HTTP endpoint a frontend can call.
