The Chrome extension isn't connected, so I'll fetch the docs directly via web fetch instead.The docs have moved. Let me fetch from the new location.Now let me fetch the official "Thinking in LangGraph" doc and the best practices page directly:Now let me also fetch the application structure page:Here's what the official LangGraph docs actually say, synthesized directly from source:

---

**From the official LangGraph docs (docs.langchain.com)**

---

**The Five-Step Process ("Thinking in LangGraph")**

The docs prescribe a concrete mental model: break the process into discrete steps (nodes), describe decisions between them (edges), connect everything through shared state. This isn't abstract — it's a literal five-step process.

**Step 1 — Map your workflow as discrete steps.** Every distinct operation becomes a node. Some nodes make decisions about where to go next (like `Classify Intent` or `Draft Reply`), while others always proceed to the same next step (like `Read Email`, which always goes to `Classify Intent`). Identify this distinction early — it determines which nodes need conditional edges and which use simple edges.

**Step 2 — Identify what each step needs.** The docs break nodes into four types: LLM steps (for reasoning, analysis, generation), data steps (for retrieval from external sources), action steps (for performing external actions like sending emails or creating tickets), and user input steps (for human intervention). For each node, you specify what static context goes in the prompt, what dynamic context comes from state, what retry strategy applies, and what the desired output is.

**Step 3 — Design your state.** State is the shared memory accessible to all nodes. A key principle: your state should store raw data, not formatted text. Format prompts inside nodes when you need them. This separation means different nodes can format the same data differently for their needs, you can change prompt templates without modifying your state schema, debugging is clearer, and your agent can evolve without breaking existing state. The rule for what goes in state: if data needs to persist across steps, put it in state. If it can be derived from other data, compute it when needed instead.

**Step 4 — Build nodes as pure functions.** A node in LangGraph is just a Python function that takes the current state and returns updates to it. The docs specify four error handling strategies based on who can fix the error: transient failures (network, rate limits) get retry policies with `RetryPolicy`; LLM-recoverable errors (tool failures, parsing issues) get stored in state so the LLM can loop back and adjust; user-fixable errors (missing info, unclear instructions) get paused with `interrupt()`; unexpected errors should bubble up for debugging — don't catch what you can't handle.

**Step 5 — Wire it together.** Graph assembly is purely structural — nodes, edges, compilation. No logic lives here.

---

**Official File Structure (from Application Structure docs)**

The docs prescribe this layout explicitly:

```
my-app/
├── my_agent/
│   ├── utils/
│   │   ├── tools.py      # tools for your graph
│   │   ├── nodes.py      # node functions
│   │   └── state.py      # state definition
│   └── agent.py          # graph construction
├── .env
├── requirements.txt
└── langgraph.json         # LangGraph config
```

The `langgraph.json` file specifies the dependencies, graphs, environment variables, and other settings. Each graph is identified by a unique name and a path to either the compiled graph or a function that makes a graph.

---

**Key Principles Stated Directly by the Docs**

Each node does one thing well. This decomposition enables streaming progress updates, durable execution that can pause and resume, and clear debugging since you can inspect state between steps. Store raw data, not formatted text — this lets different nodes use the same information in different ways. Transient failures get retries, LLM-recoverable errors loop back with context, user-fixable problems pause for input, and unexpected errors bubble up for debugging.

On node granularity: LangGraph's durable execution creates checkpoints at node boundaries. When a workflow resumes after an interruption or failure, it starts from the beginning of the node where execution stopped. Smaller nodes mean more frequent checkpoints, which means less work to repeat if something goes wrong. Isolation of external services — nodes calling external APIs should be separate because they call external services. Reusability and testing — smaller nodes are easier to test in isolation and reuse in other workflows. More nodes doesn't mean slower execution: LangGraph writes checkpoints in the background by default (async durability mode), so your graph continues running without waiting for checkpoints to complete.

---

**Production Checklist (from the docs)**

A resilient LangGraph application looks like this: State that is small, typed, and validated with reducers used sparingly. Flow using simple edges where possible and conditional edges only at real decision points with bounded cycles. Memory using a Postgres checkpointer with thread-scoped checkpoints and namespaced long-term preferences. Errors handled at node, graph, and app level with graceful degradation and escalation. Human-in-the-loop with precise interrupt points and deterministic resume paths. Operations using environment-based config, full tracing, connection pooling, and cost monitoring.
