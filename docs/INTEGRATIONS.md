# Framework Integrations

Adapters for the two most common agent frameworks. Both are thin wrappers around `MemoryStore` — you keep full access to the underlying store for advanced features.

## Installation

```bash
pip install "agent-memory-core[langchain]"    # LangChain adapter
pip install "agent-memory-core[llamaindex]"   # LlamaIndex adapter
```

## LangChain

```python
from langchain.agents import initialize_agent
from langchain_openai import ChatOpenAI
from agent_memory_core.integrations.langchain import AgentMemoryStore

memory = AgentMemoryStore(agent="support-bot")
agent = initialize_agent(
    tools=[...],
    llm=ChatOpenAI(),
    memory=memory,
    agent="conversational-react-description",
)

agent.run("What was the API key again?")
# Uses agent-memory-core's full retrieval pipeline — salience, consolidation,
# entity graph — not just cosine similarity.
```

The adapter implements LangChain's `BaseChatMemory` interface. It stores every turn as a `session`-type chunk and retrieves the top-k relevant chunks for each new input.

**Access the underlying store:**

```python
memory.store.add("API key rotated on 2026-03-15", type="credential")
report = memory.store.search("api key", n=5)
```

## LlamaIndex

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import OpenAI
from agent_memory_core.integrations.llamaindex import AgentMemoryStore

memory = AgentMemoryStore(agent="research-assistant")
agent = ReActAgent.from_tools(
    tools=[...],
    llm=OpenAI(model="gpt-5"),
    memory=memory,
)

response = agent.chat("What did we decide about the architecture refactor?")
```

Implements LlamaIndex's `BaseMemory`. Works with `ReActAgent`, `OpenAIAgent`, and any agent that accepts a `BaseMemory`.

## Sharing a store across multiple agents

Instantiate `MemoryStore` once, pass it to every adapter. Use the `agent` namespace to keep each agent's writes separate while allowing shared retrieval:

```python
from agent_memory_core import MemoryStore
from agent_memory_core.integrations.langchain import AgentMemoryStore as LCMemory
from agent_memory_core.integrations.llamaindex import AgentMemoryStore as LIMemory

store = MemoryStore()  # single backing store

# Add shared org-wide facts once
store.add("Project uses Python 3.12 + uv", type="technical")
store.add("Production DB is at db.internal:5432", type="credential")

# Per-agent adapters — each has its own namespace for session chunks
lc_mem = LCMemory(store=store, agent="support-bot")
li_mem = LIMemory(store=store, agent="research-assistant")

# Each agent's session history stays private; shared facts are visible to both
```

## Beyond adapters: direct usage

Adapters prioritize ergonomics over feature depth. For access to typed chunks, eval runs, consolidation control, replay, and custom ranking, use `MemoryStore` directly:

```python
from agent_memory_core import MemoryStore

store = MemoryStore()
store.add("First policy violation — ignore_auth header reviewed and approved", type="lesson")
store.add("User prefers 5pm standups", type="personal")

# Adaptive query intent detection picks the right ranking strategy
results = store.search("when does user want standups", n=3)
# → "personal" type + recency-heavy ranking
```

The OSS library is the full product. Adapters are just convenient glue.
