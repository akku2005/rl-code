# First-Time Historical Training Pipeline Changes

This README explains what changed in the notebooks for the first-time SageMaker run.

The important decision: this pipeline now supports first-time training from historical `campaign_sent` events without requiring production decision logs.

## Why This Change Was Needed

Before these changes, the pipeline mixed two different modes:

- production replay mode, which requires real decision logs, logged candidate slates, and logged propensities
- first-time historical mode, where only historical campaign events exist

For a first-time model, production decision logs do not exist yet. Industry practice is to bootstrap with historical data, clearly mark the propensities as synthetic, train the first model, and keep it non-production-promotable until real logged propensities are collected.

## Files Changed

| File | Before | After |
|---|---|---|
| `0003_build_reward_engine_FIXED.ipynb` | Used older reward logic and stale `vw_cost = -net_reward` assumptions | Uses historical `campaign_sent` rows, attributes outcomes, computes bounded `vw_cost` in `[0, 1]` |
| `04_Task4_FINAL.ipynb` | Required decision-log style columns such as `decision_id`, `decision_ts`, `candidate_slate_json`; also carried `frequency` and `creative` | Generates first-time `decision_id`, 60-action slates, and uniform synthetic propensity using only `channel`, `day`, `offer`, and `time` |
| `5_build_vw_formatter.ipynb` | Required strict logged slates and true logged propensities | Accepts `cold_start_uniform` mode and writes VW ADF blocks without `frequency` or `creative` action features |
| `6_train_vw_model.ipynb` | Required `strict_ope` only | Allows `cold_start_uniform`, but keeps `production_promotable = False` |
| `2_generate_action_library.ipynb` | Had hardcoded paths and silent upload failure paths | Uses configurable bucket and raises upload/load failures |
| `7_evaluate_vw_model.ipynb` | Mostly unchanged | Keeps production gates strict |
| `8_daily_scoring_pipeline.ipynb` | Mostly unchanged | Still creates real decision logs after model scoring |

## Before: Old First-Run Failure Pattern

### Historical Reward Label Example

Old Task 3 could produce a campaign-level reward table like this:

| campaign_id | master_user_id | campaign_timestamp | channel_id | net_reward | vw_cost |
|---|---|---|---|---:|---:|
| c1 | u1 | 2025-01-01 08:00:00 | email | 8.95 | -8.95 |
| c2 | u2 | 2025-01-01 09:00:00 | sms | -0.12 | 0.12 |

Problem:

- VW cost was based on `vw_cost = -net_reward`.
- Positive business reward became negative VW cost.
- Downstream notebooks validate `vw_cost` must be in `[0, 1]`.
- A row like `vw_cost = -8.95` is invalid for the current training contract.

### Task 4 Expected Decision-Log Columns

Before, Task 4 expected columns like this:

| decision_id | decision_ts | action_id | chosen_action_prob | candidate_slate_json | vw_cost |
|---|---|---:|---:|---|---:|
| d1 | 2025-01-01 08:00:00 | 104 | 0.92 | `[104,205,...]` | 0.21 |

But first-time historical reward labels usually looked like this:

| campaign_id | master_user_id | campaign_timestamp | channel_id | vw_cost |
|---|---|---|---|---:|
| c1 | u1 | 2025-01-01 08:00:00 | email | 0.49 |
| c2 | u2 | 2025-01-01 09:00:00 | sms | 0.95 |

Problem:

- no `decision_id`
- no `decision_ts`
- no `candidate_slate_json`
- no real `chosen_action_prob`
- no logged candidate slate

So Task 4 and Task 5 were correct for production replay, but not runnable for a first-time historical bootstrap.

## After: New First-Time Historical Path

### Task 3 Output After Change

Task 3 now uses historical `campaign_sent` rows as the anchor and keeps all sends, including zero-outcome sends.

Example:

| campaign_id | master_user_id | campaign_timestamp | channel_id | total_event_reward | channel_cost | net_reward | vw_cost |
|---|---|---|---|---:|---:|---:|---:|
| c1 | u1 | 2025-01-01 08:00:00 | email | 10.00 | 0.05 | 8.95 | 0.494256 |
| c2 | u2 | 2025-01-01 09:00:00 | sms | 0.00 | 0.12 | -0.12 | 0.948253 |

The formula is:

```text
net_reward = decayed_reward - channel_cost
```

Then Task 3 converts the reward to a bounded VW cost:

```text
bounded_reward = clip(net_reward, reward_floor, reward_ceiling)
scaled_reward = log1p(bounded_reward - reward_floor) / log1p(reward_ceiling - reward_floor)
vw_cost = 1 - scaled_reward
```

Result:

- higher business reward becomes lower VW cost
- worse outcome becomes higher VW cost
- all costs stay in `[0, 1]`

### Task 4 Output After Change

Task 4 now creates a first-time historical training table.

Example:

| decision_id | campaign_id | master_user_id | action_id | channel | day | offer | time | vw_cost | chosen_action_prob | slate_size | decision_policy | propensity_source |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---|---|
| hist::c1::u1::20250101T080000::00000000 | c1 | u1 | 4 | CH005 | DW001 | OF001 | TM001 | 0.494256 | 0.016667 | 60 | cold_start_uniform | synthetic_uniform_slate |
| hist::c2::u2::20250101T090000::00000001 | c2 | u2 | 1 | CH002 | DW001 | OF001 | TM001 | 0.948253 | 0.016667 | 60 | cold_start_uniform | synthetic_uniform_slate |

Example generated slate:

```json
[4, 60, 67, 34, 19, 39, 37, 31, 45, 33, 14, 23, 59, 50, 18, 0, 36, 3, 51, 27, 9, 28, 61, 7, 11, 16, 68, 52, 53, 54, 24, 55, 64, 44, 6, 56, 46, 43, 62, 48, 2, 17, 5, 66, 32, 49, 29, 10, 40, 65, 35, 13, 21, 20, 42, 8, 12, 38, 1, 57]
```

The selected historical action is always included in the slate.

`frequency` and `creative` are not used as model action features in the first-time path. If the action library contains many rows that differ only by those fields, Task 4 chooses the lowest `action_id` as the canonical representative for each `channel + day + offer + time` combination.

### Task 5 VW Block After Change

Task 5 writes VW ADF format.

Example:

```text
shared |emb v0:0.0 v1:0.01 ... |user inc=mid city=tier1 life=active risk=low cibil:720 products:1
0:0.4943:0.016667 |action aid_4 ch=CH005 dw=DW001 tm=TM001 of=OF001
|action aid_60 ch=CH001 dw=DW001 tm=TM001 of=OF001
|action aid_67 ch=CH003 dw=DW001 tm=TM002 of=OF002
...
```

Important:

- `0.4943` is the bounded VW cost.
- `0.016667` is synthetic uniform propensity, equal to `1 / 60`.
- This is useful for bootstrapping, but it is not true production logged propensity.

## Training and Validation Meaning

After the change:

| Stage | Can run first time? | Production-promotable? | Reason |
|---|---:|---:|---|
| Task 3 reward labels | Yes | No | Historical reward attribution only |
| Task 4 unified table | Yes | No | Synthetic slates and propensities |
| Task 5 VW formatter | Yes | No | `strict_ope_ready = False` |
| Task 6 training | Yes | No | `cold_start_uniform` bootstrap model |
| Task 7 evaluation | Yes | Usually no | OPE is not reliable without true logged propensities |
| Task 8 scoring | Shadow only unless gates pass | No for first run | Needs real evaluation gates and future logged decisions |

## Correct Industry-Practice Interpretation

The first model can be trained from historical data, but it should not be treated as fully validated production policy learning.

The correct flow is:

```text
historical events
-> Task 3 reward labels
-> Task 4 cold-start slates
-> Task 5 VW ADF training file
-> Task 6 bootstrap model
-> Task 7 diagnostic evaluation
-> Task 8 shadow scoring
-> collect real decision logs and outcomes
-> rerun Task 3 onward in strict OPE mode
```

## Outputs To Check On SageMaker

After running the notebooks, check these S3 outputs:

```text
s3://aks-nvtabular-data/action_library/action_library_pl-aip-uplift.parquet
s3://aks-nvtabular-data/rewards/reward_labels_pl-aip-uplift_LOG_SCALED.parquet
s3://aks-nvtabular-data/rewards/event_audit_pl-aip-uplift.parquet
s3://aks-nvtabular-data/training_data/unified_training_pl-aip-uplift.parquet
s3://aks-nvtabular-data/training_data/vw_training_pl-aip-uplift_FINAL.txt
s3://aks-nvtabular-data/training_data/vw_propensity_audit_pl-aip-uplift.json
s3://aks-nvtabular-data/model_artifacts/pl-aip-uplift/latest_run.json
```

Expected audit flags for the first run:

```json
{
  "training_mode": "cold_start_uniform",
  "has_true_logged_propensity": false,
  "has_non_logged_propensity": true,
  "strict_ope_ready": false,
  "production_promotable": false
}
```

