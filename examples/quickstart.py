"""
Quickstart — agent-memory-core in 3 lines.

Run:
    pip install agent-memory-core
    python examples/quickstart.py
"""

from agent_memory_core import MemoryStore

store = MemoryStore()
store.add("The API key is stored in the keychain", type="credential")
store.add("The project uses Python 3.12", type="technical")

results = store.search("Where is the API key?")
print(results[0].text)  # "The API key is stored in the keychain"
