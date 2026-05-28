# SI Model Integration Report
## Cascading Failure Model for Flood Simulator

**Reference:** SI model adapted from MATLAB implementation (SCCIFMI.m)

---

## 1. Overview

This report documents the integration of a Susceptible-Infected (SI) cascading failure model into the gym_style flood simulator. The SI model extends the original direct-only failure mechanism by adding network-based cascading failures through power line topology.

### Purpose
- **Original Model (IID):** Failures occur only through direct flood exposure
- **SI Model:** Failures occur through both direct exposure AND cascading through network connections
- **Flexibility:** Seamless toggle between SI and IID models via `use_si_model` parameter

---

## 2. Theoretical Framework

### 2.1 Direct Failure Probability
Both models use the same lognormal fragility function:

```
P_direct(x) = 0.5 * erfc(-(log(x) - μ) / (σ * √2))
```

Where:
- `x` = flood depth at line location (meters)
- `μ` = ln-mean parameter (-0.22, calibrated to flood depth)
- `σ` = ln-std parameter (0.30)
- `erfc()` = complementary error function

**Interpretation:** Probability that a line fails given flood depth `x`.

### 2.2 Cascading Failure Probability (SI Model Only)

For each intact line `j` at time `t`, the cascading failure probability is:

```
P_cascade(j,t) = 1 - exp( Σ_i log(1 - q(i,j) * A(i,j) * [c(t,i) == 0]) )
```

Where:
- `q(i,j)` = transmission probability from failed line `i` to intact line `j`
  - Currently: `q(i,j) = edge_factor` (fixed value for all edges)
  - Physical meaning: probability that a failed neighbour causes line `j` to also fail
- `A(i,j)` = adjacency matrix (1 if lines share a substation node, 0 otherwise)
- `c(t,i) == 0` = line `i` is already failed at time `t`
- `edge_factor` = user-tuned parameter (0–1) controlling cascade transmission strength

**Key Properties:**
- Uses log-space arithmetic for numerical stability
- Only failed neighbours contribute (`c(t,i) == 0`)
- `q` is currently a fixed scalar — see Section 8.2 for how to extend it

### 2.3 Two-Pass Failure Mechanism

Direct and cascade failures are evaluated as **two independent passes** each timestep:

**Pass 1 — Direct (flood zone only):**
```
for each line j currently flooded:
    if j is intact:
        p_direct = lognormal_cdf(local_depth_j)
        if random() < p_direct → mark j failed
```

**Pass 2 — Cascade (all intact lines, including outside flood zone):**
```
for each line j still intact:
    p_cascade = 1 - exp(Σ log(1 - q * A[i,j])) over all failed neighbours i
    if random() < p_cascade → mark j failed
```

Pass 2 is skipped entirely when `use_si_model=False` (IID mode).

**Important:** A line outside the flood zone can fail via cascade (Pass 2) even though it would never fail from direct flood stress (Pass 1). This is the key difference between IID and SI.

### 2.4 Network Topology (Adjacency Matrix)

Two power lines are considered adjacent if they share a substation node:
- Line `i` and Line `j` are connected if:
  - `from_node[i]` ∈ {from_node[j], to_node[j]} OR
  - `to_node[i]` ∈ {from_node[j], to_node[j]}

This is bidirectional (undirected graph).

### 2.5 Adaptation from MATLAB Original

The original MATLAB model (SCCIFMI.m) was designed for **windstorm** damage on a node-based network. Key differences in this Python flood adaptation:

| Aspect | MATLAB (wind) | Python (flood) |
|---|---|---|
| Hazard type | Wind speed | Flood depth |
| Model unit | Nodes (substations) | Edges (power lines) |
| Direct fragility | Piecewise linear in wind speed | Lognormal CDF in flood depth |
| `q` formula | Depends on edge length + wind intensity | Fixed `edge_factor` |
| Line types | OH (overhead) vs UG (underground) | Single type |
| Cascade trigger | Only **newly-failed** nodes (`c(t,i)*(1-c(t-1,i))`) | All failed lines each timestep |

The fixed `edge_factor` is a deliberate simplification: flood-driven cascade transmission lacks published calibration data, so a tunable scalar is used instead of a length/intensity-dependent formula. To extend `q` in future, see Section 8.2.

---

## 3. Files Modified

### 3.1 `gym_style/algo2_powerline.py`

**Changes Made:**

#### a) Constructor `__init__()` — Added parameters
```python
use_si_model=False,      # Toggle SI cascading model
edge_factor=0.5,         # Cascade transmission probability per edge (0-1)
```

#### b) New Method: `_build_adjacency_matrix()` (~20 lines)
- Constructs N×N binary adjacency matrix from power line topology
- Reads `from_node` and `to_node` columns from GeoDataFrame
- Marks lines as connected if they share a node

#### c) New Method: `_compute_cascading_probability(line_idx)` (~15 lines)
- Computes P_cascade using log-space arithmetic
- Iterates over all failed neighbours of `line_idx`
- Uses fixed `q = edge_factor` per edge

#### d) Modified `step()` — Two-pass structure
```python
# Pass 1: direct failures (flooded lines only)
for i in flooded_lines:
    if intact: draw against p_direct

# Pass 2: cascade failures (ALL intact lines)
if use_si_model:
    for i in all_intact_lines:
        draw against p_cascade
```

---

### 3.2 `gym_style/flood_env.py`

**Changes Made:**

#### a) Constructor `__init__()` — Added 2 parameters
```python
use_si_model=False,      # Passed to PowerlineFailureEnv
edge_factor=0.5,         # Passed to PowerlineFailureEnv
```

#### b) PowerlineFailureEnv instantiation
```python
self._algo2 = PowerlineFailureEnv(
    ...,
    use_si_model=use_si_model,
    edge_factor=edge_factor,
)
```

---

### 3.3 `gym_style/compare.py` (New File)

Side-by-side visualisation of IID vs SI across all three infrastructure layers (power lines, roads, telecom towers). Runs both models in parallel with the same flood data and seed, stepping manually via Enter key.

```
python gym_style/compare.py
```

Parameters (top of `main()`):
- `edge_factor = 0.8`
- `mu = -0.22` (paper default)
- `seed = 42`

---

### 3.4 `flood_sim/algo2_powerline_failure_si.py` (Reference)

Standalone batch implementation. Not used by the gym environments directly — kept as a reference.

---

## 4. Integration Architecture

```
FloodDisasterEnv (gym_style/flood_env.py)
  └─ PowerlineFailureEnv (gym_style/algo2_powerline.py)
     ├─ use_si_model: bool
     ├─ edge_factor: float
     ├─ _adjacency: np.ndarray (N×N)
     └─ step() → Pass 1 (direct) + Pass 2 (cascade, SI only)
  └─ RoadBlockageEnv   — IID only, no cascade
  └─ TelecomFailureEnv — no fragility; fails when substation loses power
```

Roads (`algo3`) have no cascade mechanism — a blocked road does not cause adjacent roads to block. Telecom towers (`algo4`) do not use SI directly; they inherit the effect through `algo2`'s `L_status`.

---

## 5. Usage Guide

### 5.1 Basic Usage

```python
from gym_style.algo2_powerline import PowerlineFailureEnv

env = PowerlineFailureEnv(use_si_model=True, edge_factor=0.6)
obs, _ = env.reset(seed=42)

terminated = False
while not terminated:
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"Lines failed: {info['lines_failed']}")
```

### 5.2 Side-by-Side Comparison

```bash
python gym_style/compare.py
```

Produces a 2-panel animated plot: IID (left) vs SI (right), showing power lines, roads, and telecom towers. Press Enter to advance each hour.

### 5.3 Comparing Models Programmatically

```python
env_iid    = PowerlineFailureEnv(use_si_model=False)
env_si_low = PowerlineFailureEnv(use_si_model=True, edge_factor=0.2)
env_si_med = PowerlineFailureEnv(use_si_model=True, edge_factor=0.5)
env_si_hi  = PowerlineFailureEnv(use_si_model=True, edge_factor=0.9)
```

---

## 6. Parameter Reference

### 6.1 SI-Specific Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `use_si_model` | bool | False | True/False | Enable SI cascading model |
| `edge_factor` | float | 0.5 | 0.0–1.0 | Per-edge transmission probability `q` |

### 6.2 Fragility Parameters (Unchanged)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mu` | float | -0.22 | Lognormal ln-mean; median failure at exp(μ) metres |
| `sigma` | float | 0.30 | Lognormal ln-std |

### 6.3 Edge Factor Interpretation

`edge_factor` is the probability that a single failed neighbour causes an intact adjacent line to fail:

```
edge_factor = 0.0  → No cascading (identical to IID)
edge_factor = 0.2  → Weak cascade   — 20% transmission per failed neighbour
edge_factor = 0.5  → Medium cascade — 50% transmission per failed neighbour
edge_factor = 0.9  → Strong cascade — 90% transmission per failed neighbour
```

If a line has multiple failed neighbours, probabilities combine via the product formula (P_cascade increases with each additional failed neighbour).

---

## 7. Algorithm Comparison

### 7.1 IID Model

```
Each timestep:
  for each flooded intact line j:
    P_fail = lognormal_cdf(depth_j)
    if random() < P_fail → fail
```

- ✅ Simple, O(T×N)
- ❌ Ignores network topology
- ❌ Cannot propagate failures outside flood zone

### 7.2 SI Model

```
Each timestep:
  Pass 1 — for each flooded intact line j:
    P_direct = lognormal_cdf(depth_j)
    if random() < P_direct → fail

  Pass 2 — for each intact line j (including outside flood):
    P_cascade = 1 - exp(Σ log(1 - edge_factor * A[i,j])) over failed i
    if random() < P_cascade → fail
```

- ✅ Models network topology
- ✅ Failures can propagate outside flood zone
- ✅ More failures earlier (faster degradation)
- ⚠️ O(T×N²) per episode

### 7.3 Observed Output (seed=42, edge_factor=0.8, mu=-0.22)

| Metric | IID | SI |
|---|---|---|
| Lines failed (final) | 136 / 143 | 143 / 143 |
| Cascade-only failures | 0 | +7 |
| Towers failed (final) | 125 / 164 | 164 / 164 |
| Tower difference | — | +39 via cascade |

The cascade effect is most visible mid-simulation (e.g. Hour 13: 5 vs 19 lines failed). By Hour 23 the flood covers the entire network so both models converge toward full failure. The SI model causes failures **earlier and faster**.

---

## 8. Implementation Details

### 8.1 Adjacency Matrix Construction

```python
def _build_adjacency_matrix():
    A = N×N zero matrix
    for i in 0..N-1:
        for j in 0..N-1 (j != i):
            if lines[i].from_node or to_node matches lines[j].from_node or to_node:
                A[i,j] = 1
    return A  # symmetric, zero diagonal
```

Built once at `__init__`, reused every timestep.

### 8.2 Cascading Probability & Extending `q`

Current implementation:
```python
def _compute_cascading_probability(self, line_idx):
    log_sum = 0.0
    for i in range(self.n_lines):
        if i != line_idx and self._L_status[i] == 0 and self._adjacency[i, line_idx]:
            q_ij = self.edge_factor          # ← change this line to extend the model
            log_sum += log(max(1.0 - q_ij, 1e-10))
    return 1.0 - np.exp(log_sum)
```

To make `q` depend on flood depth (closer to MATLAB's intensity-dependent formula):
```python
q_ij = self.edge_factor * local_depth / max_depth
```

Only this one line needs changing — all surrounding logic stays the same.

### 8.3 State Transition

```
t=0: all lines intact

for each timestep t:
    Pass 1 (direct):
        newly in flood zone this step → draw against p_direct
        failed lines stay failed forever

    Pass 2 (cascade, SI only):
        all still-intact lines → draw against p_cascade
        failed lines stay failed forever

    t += 1
```

No repair or recovery is modelled (S→I only, no R state).

---

<!-- ## 9. Performance

| Operation | IID | SI |
|---|---|---|
| Build adjacency | N/A | O(N²) once |
| Per timestep | O(N) | O(N²) |
| Typical runtime (T=24, N=143) | <1ms | <10ms |
| Memory (adjacency) | N/A | ~160KB |

---

## 10. Future Extensions

1. **Depth-dependent `q`** — replace fixed `edge_factor` with `f(flood_depth, edge_length)`
2. **Line type distinction** — overhead vs underground lines with different `q`
3. **Sparse adjacency** — use `scipy.sparse` if N > 1000
4. **Recovery/repair** — add R state (SIR model) with time-to-repair distribution

--- -->

## 9. References

- **MATLAB Source:** `SCC IMFI Files/SCC IMFI Files/IMFI Files MATLAB/SCCIFMI.m`
- **Lognormal Fragility:** Standard in flood/wind/seismic engineering literature
- **SI Model:** Adapted from epidemiological propagation models
- **Network Cascading:** Common in power grid reliability analysis

---

**End of Report**

Implementation: `gym_style/algo2_powerline.py`  
Visualisation: `gym_style/compare.py`  
Reference (batch): `flood_sim/algo2_powerline_failure_si.py`
