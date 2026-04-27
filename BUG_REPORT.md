# VW Contextual Bandit Pipeline — Bug Report & Fixes

**Use Case:** pl-aip-uplift (Personal Loan AIP Uplift)  
**Notebooks Audited:** Task 5, Task 6, Task 7  
**Total Bugs Found:** 10  

---

## Bug Summary Table

| # | Notebook | Severity | Bug | Impact |
|---|----------|----------|-----|--------|
| 1 | Task 6 | 🔴 CRASH | `vw.learn(lines)` — list[str] passed instead of block string | RuntimeError crash on every training block |
| 2 | Task 6 | 🔴 CRASH | `vw.predict(stripped)` — list[str] passed; tuple output parsed with `np.argmax` | Wrong predictions + crash |
| 3 | Task 6 | 🔴 WRONG PARSE | LABEL_RE does not match Task 5 output format | All block parses return None → eval crashes |
| 4 | Task 6 | 🟠 WRONG MODE | `--cb_adf` used instead of `--cb_explore_adf` for training | cb_type sweep (dr/mtr) has no effect |
| 5 | Task 6 | 🟡 BIASED METRIC | Train loss proxy sampled only 500 blocks | Biased train loss estimate for small datasets |
| 6 | Task 5 | 🔴 WRONG FORMAT | Label written as `{cost}:{prob}` missing required `0:` action-index prefix | VW rejects the example: "only one line can have a cost" |
| 7 | Task 5 | 🟠 MISMATCH | LABEL_RE in Task 5 validates without `0:` but Task 6/7 expect it | Cross-notebook format inconsistency |
| 8 | Task 7 | 🟠 WRONG LOGIC | LABEL_RE captures `action_idx` and uses it as `chosen_idx` | `action_idx` is always `0` in cb_adf (not a position) |
| 9 | Task 7 | 🟡 BRITTLE | `strip_labels` regex pattern inconsistent with LABEL_RE | Fails silently if format varies |
| 10 | Task 7 | 🟡 FRAGILE | `parse_pred_token` doesn't handle `action_idx:prob` VW output format robustly | Wrong slate positions used in eval |

---

## Detailed Bug Analysis

---

### BUG 1 — Task 6 `train_candidate()`: `vw.learn(lines)` receives a list[str]

**Location:** `6_train_vw_model_FIXED.ipynb` → `train_candidate()` function

**Original code:**
```python
for block in blocks:
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    if lines:
        vw.learn(lines)   # ← WRONG: list[str] passed
```

**Error produced:**
```
RuntimeError: cb_adf: badly formatted example, only one line can have a cost
```

**Root cause:**  
The VW Python API for `--cb_adf` / `--cb_explore_adf` requires the **entire multiline block as a single string** with embedded newlines. Each "example" spans multiple lines (shared context line + one line per action). Passing a `list[str]` causes VW to interpret each string as a separate single-line example, each of which contains a label — violating the "only one label per multiline example" rule.

**Fix:**
```python
for block in blocks:
    vw.learn(block)   # ✅ pass the whole block string
```

**VW API reference:**  
From VW wiki: *"Each example now spans multiple lines, with one line per action. A new line signals end of a multiline example."*

---

### BUG 2 — Task 6 `evaluate_candidate()`: `vw.predict()` receives list[str]; argmax applied to tuples

**Location:** `6_train_vw_model_FIXED.ipynb` → `evaluate_candidate()` and `strip_labels_from_block()`

**Original code:**
```python
def strip_labels_from_block(block: str) -> list[str]:  # returns list
    out = []
    ...
    return out  # ← list[str]

# Then:
scores = vw_eval.predict(stripped)   # stripped is list[str] → wrong
best_pos = int(np.argmax(scores))    # np.argmax on tuples → wrong
```

**Two sub-bugs:**
1. `predict()` receives `list[str]` but needs a single block string (same as `learn()`)
2. For `--cb_explore_adf`, `predict()` returns `list[(action_idx, prob)]` tuples sorted by probability descending — `np.argmax` on tuples does lexicographic comparison, not probability comparison

**Fix:**
```python
def strip_labels_from_block(block: str) -> str:   # returns string
    ...
    return '\n'.join(out)   # ✅ joined string

# Then:
pred_result = vw_eval.predict(stripped_block)    # string input
if isinstance(pred_result[0], tuple):
    best_pos = int(pred_result[0][0]) - 1        # 1-based action idx → 0-based position
```

---

### BUG 3 — Task 6 LABEL_RE does not match Task 5's output format

**Location:** `6_train_vw_model_FIXED.ipynb` → `LABEL_RE` constant

**Task 5 writes:**
```
0:0.3421:0.016667 |action aid_42 ch=CH001 dw=DW002 ...
```

**Task 6 v1 LABEL_RE (broken):**
```python
LABEL_RE = re.compile(r'^\s*(?P<cost>[0-9]*\.?[0-9]+):(?P<prob>[0-9]*\.?[0-9]+)\s+\|action\b')
```
This regex does NOT have `0:` before cost, so it matches the format `cost:prob |action` but Task 5 outputs `0:cost:prob |action`. Every block parse returns `None` for cost/reward/prob.

**Fix:**
```python
LABEL_RE = re.compile(r'^\s*(?:0:)?(?P<cost>[0-9]*\.?[0-9]+):(?P<prob>[0-9]*\.?[0-9]+)\s+\|action\b')
# (?:0:)? makes the action-index prefix optional — accepts both formats
```

---

### BUG 4 — Task 6: `--cb_adf` used instead of `--cb_explore_adf` for training

**Location:** `6_train_vw_model_FIXED.ipynb` → `build_vw_args()`

**Original:**
```python
'--cb_adf',
```

**Problem:**  
`--cb_adf` is the *offline-only* contextual bandit reduction. It does not register the exploration-specific `--cb_type` reduction correctly. The sweep over `cb_type in ('dr', 'mtr')` only takes effect under `--cb_explore_adf`. With `--cb_adf`, both candidates behave identically.

Additionally, the saved model file needs to be loaded with matching flags during evaluation (`--cb_explore_adf -t`).

**Fix:**
```python
'--cb_explore_adf',
'--epsilon 0.0',   # pure exploitation during offline replay training
```

---

### BUG 5 — Task 6: Train loss proxy uses only 500 blocks

**Location:** `6_train_vw_model_FIXED.ipynb` → `train_candidate()`

**Original (in the fallback branch):**
```python
if total_loss == 0.0 and n > 0:
    costs = []
    for block in blocks[:min(500, len(blocks))]:  # ← only 500 blocks sampled
```

**Problem:**  
For cold-start with few users, 500 blocks can be the entire dataset or even more than available. The sample bias is unpredictable. More critically, the cost collection is outside the training loop, meaning it requires a second pass over the data.

**Fix:** Collect costs *during* the training loop (one pass) over all blocks:
```python
for block in blocks:
    vw.learn(block)
    for line in block.splitlines():
        m = LABEL_RE.match(line.strip())
        if m:
            costs.append(float(m.group('cost')))
            break
avg_loss = float(np.mean(costs)) if costs else 0.5
```

---

### BUG 6 — Task 5: VW label missing required `0:` action-index prefix

**Location:** `5_build_vw_formatter.ipynb` → Cell 3 (VW block generation)

**Original code:**
```python
block_lines.append(
    f"{cost}:{prob:.6f} |action aid_{taken_action} ..."   # ← missing '0:'
)
```

**VW cb_adf canonical format:**
```
0:{cost}:{prob} |action features...
```
The `0` is the action index (always 0 in cb_adf format — VW ignores it but requires it syntactically). Without it, VW cannot parse the line as a labeled action and raises:
```
RuntimeError: cb_adf: badly formatted example, only one line can have a cost
```

**Fix:**
```python
block_lines.append(
    f"0:{cost}:{prob:.6f} |action aid_{taken_action} ..."   # ✅ with '0:'
)
```

**VW format reference:**  
Per VW wiki: *"For each action, we have the label information (action, cost, probability), if known. The action field `a` is ignored now since line numbers identify actions and is typically set to 0."*

---

### BUG 7 — Task 5: LABEL_RE validation regex does not require `0:` prefix

**Location:** `5_build_vw_formatter.ipynb` → Cell 3

**Original:**
```python
LABEL_RE = re.compile(r"^0:(?P<cost>[0-9]*\.?[0-9]+):(?P<prob>[0-9]*\.?[0-9]+)\s+\|action\b")
```
(This version in Task 5 actually *does* have `0:` — so the validation regex is correct, but the *generation* code in the same cell was missing it. The inconsistency means a block could be written without `0:` but the validator would reject it at generation time — a silent logic failure.)

**Fix:** Ensure both the generation AND the LABEL_RE are consistent with `0:cost:prob |action` format. The canonical LABEL_RE for all notebooks should be:
```python
LABEL_RE = re.compile(r"^0:(?P<cost>[0-9]*\.?[0-9]+):(?P<prob>[0-9]*\.?[0-9]+)\s+\|action\b")
```

---

### BUG 8 — Task 7: LABEL_RE captures `action_idx` and uses it as `chosen_idx`

**Location:** `7_evaluate_vw_model.ipynb` → `LABEL_RE` and `parse_block()`

**Original:**
```python
LABEL_RE = re.compile(
    r"^\s*(?P<action_idx>[0-9]+):(?P<cost>...):(?P<prob>...)\s+\|action\b"
)
# In parse_block():
chosen_idx = int(m.group("action_idx"))  # always 0 — not the slate position
r["chosen_idx"] = chosen_idx
```

**Problem:**  
`action_idx` in the cb_adf label format is **always 0** (VW ignores it for position determination). Using it as the slate position always produces `chosen_idx=0`, which happens to be correct (since we always write the chosen action first) — but for the **wrong reason**. The code is semantically incorrect and would silently break if the ordering assumption ever changes.

**Fix:** Remove the `action_idx` named group; set `chosen_idx = i` (actual line position):
```python
LABEL_RE = re.compile(r"^\s*0:(?P<cost>...):(?P<prob>...)\s+\|action\b")
# In parse_block():
r["chosen_idx"] = i    # ✅ actual line position (always 0 in our format)
```

---

### BUG 9 — Task 7: `strip_labels()` returns string but downstream code may expect list

**Location:** `7_evaluate_vw_model.ipynb` → `strip_labels()`

**Original:**
```python
def strip_labels(block):
    out = []
    for line in block.splitlines():
        ...
        out.append(s)
    return "\n".join(out)   # returns string — this part is OK
```

The function returns a string correctly, but the stripping regex `r"^\s*[0-9]+:[0-9.]+:[0-9.]+\s+"` uses `[0-9]+` which requires at least one digit — correct. However it doesn't account for the `0:` being the action index. We make this explicit and consistent with the fixed LABEL_RE.

**Fix:** Update strip regex to precisely match canonical format:
```python
s = re.sub(r"^\s*0:[0-9.]+:[0-9.]+\s+", "", s)
```

---

### BUG 10 — Task 7: `parse_pred_token()` fragile for `cb_explore_adf` output format

**Location:** `7_evaluate_vw_model.ipynb` → `parse_pred_token()`

**VW `--cb_explore_adf -t -p` output format:**
```
2:0.95 1:0.03 3:0.02    ← action_idx:prob pairs, first = recommended
```

**Original:**
```python
def parse_pred_token(line: str) -> int:
    token = line.split()[0].split(",")[0]
    if ":" in token:
        token = token.split(":", 1)[0]
    return int(float(token))
```
This mostly works for the `action_idx:prob` format, but fails silently when VW outputs a pure float (e.g. from `--cb_adf` models or format edge cases).

**Fix:** Add explicit handling for both formats and range validation:
```python
def parse_pred_token(line: str) -> int:
    first_token = line.split()[0]
    if ":" in first_token:
        idx_str = first_token.split(":")[0]
        try:
            return int(float(idx_str))
        except ValueError:
            return 1
    else:
        try:
            val = float(first_token)
            if val == int(val) and 1 <= int(val) <= NUM_SLATE:
                return int(val)
            return 1
        except ValueError:
            return 1
```

---

## Cross-Notebook Format Contract

All three fixed notebooks now agree on this **canonical VW cb_adf block format**:

```
shared |emb v0:0.123 v1:-0.456 ... v63:0.789 |user inc=TIER1 city=T1 life=active risk=low cibil:780 products:3
0:0.3421:0.016667 |action aid_42 ch=CH001 dw=DW002 tm=morning of=OF001
|action aid_7 ch=CH002 dw=DW003 tm=evening of=OF002
|action aid_15 ch=CH001 dw=DW001 tm=lunch of=OF003
... (60 action lines total, chosen action always first with 0:cost:prob label)
```

**LABEL_RE (unified, used in all three notebooks):**
```python
LABEL_RE = re.compile(r"^\s*(?:0:)?(?P<cost>[0-9]*\.?[0-9]+):(?P<prob>[0-9]*\.?[0-9]+)\s+\|action\b")
# (?:0:)? = optional for backward compatibility; Task 5 always writes '0:'
```

---

## Files Not Changed

- `04_Task4_FINAL.ipynb` — no bugs found; cold-start slate generation is correct
- `0003_build_reward_engine_FIXED.ipynb` — no bugs found; reward computation and vw_cost scaling is correct
- `2_generate_action_library.ipynb` — no bugs found; Cartesian product generation is correct  
- `8_daily_scoring_pipeline.ipynb` — no bugs found; uses subprocess VW (not Python API) correctly; label-free scoring file format is correct
- `action_banks_pl-aip-uplift_V2_FEATURE_BASED.json` — config only, no code bugs
- `campaign_events.json` — config only
- `funnel_config.json` — config only
