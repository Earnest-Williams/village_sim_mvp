# AGENTS.md: Repository Standards & Directives

**To any AI or LLM operating within this repository:**

You are interacting with `village_sim_mvp`, a high-throughput, emergent simulation engine. We prioritize performance, memory locality, and deterministic execution over "clever" object-oriented abstractions. 

When generating code, analyzing bugs, or refactoring, you MUST adhere to the following architecture rules:

### 1. Performance First (Vectorize or Fail)
* Never iterate over large collections (grids, agent pools) using Python `for`-loops. 
* Use **NumPy** for spatial/environmental simulation. Accelerate complex math with Numba `@njit`.
* Use **Polars** DataFrames for managing large sets of agent states/needs. Use columnar operations and `.filter()`, never `.apply()`.

### 2. Typing is Law
* This repo follows `mypy --strict`.
* Use explicit type annotations for all function signatures.
* Use Python 3.10+ syntax: `list[int]`, `dict[str, int]`, and `Type | None`. 
* The `typing` module's `Optional`, `List`, `Dict`, and `Union` are banned.

### 3. Serialization
* Use `msgpack` for all state storage, replays, and snapshotting. 
* Standard library `json` is banned due to I/O and parsing overhead. File interactions must be binary (`wb`, `rb`).

### 4. Code Style
* No unnecessary OOP. Prefer pure functions operating on data structures over methods mutating internal class state.
* Keep imports clean and alphabetical.
* No syntactic sugar at the cost of readability or performance.

If your proposed code violates these principles, rewrite it before presenting it.
