# Task 3 Reward Engine README

This file explains, in plain English, what `3_build_reward_engine_FIXED.ipynb` does and how it fits into the VW training pipeline.

## What this notebook does

Task 3 turns raw campaign outcome data into a training label that Vowpal Wabbit can use.

In simple terms:

1. Look at each campaign send.
2. Check what happened after that send.
3. Give that send a reward score.
4. Convert the reward score into a VW cost.
5. Save the result for the next notebook.

VW learns from **cost**, not from raw business reward. Lower cost is better.

## Why this notebook exists

VW contextual bandits expect training rows like this:

```text
chosen_action:cost:probability
```

So we cannot feed raw event logs directly into VW. We first need to:

- attribute outcomes to a campaign send
- calculate a net reward
- convert that reward into a cost VW can learn from

## Inputs

Task 3 reads:

- `config/reward_config.json`
- `config/funnel_config.json`
- raw event history from `full_df.parquet`
- the decision log from `8_daily_scoring_pipeline.ipynb` or the latest decision-log manifest

## What the notebook calculates

For each campaign send, Task 3 calculates:

- `events_found_all`
- `events_found_rewarding`
- `events_found_positive`
- `total_event_reward`
- `time_decay_factor`
- `decayed_reward`
- `channel_cost`
- `net_reward`
- `vw_cost`

### Plain-English meaning

- `decayed_reward` means the reward gets smaller the longer it takes to happen.
- `channel_cost` means the channel itself has a business cost.
- `net_reward` means what is left after subtracting channel cost from reward.
- `vw_cost` means the number VW will try to minimize.

## Simple example

| Campaign result | Net reward | VW cost | Meaning |
|---|---:|---:|---|
| No conversion on Push | `-0.03` | `0.96` | bad outcome |
| Small value on SMS | `25` | `0.70` | okay outcome |
| Strong value on Email | `200` | `0.20` | good outcome |
| Very large value | `5000` | `0.00` | best outcome |

The exact numbers will depend on the configured reward floor and ceiling, but the idea stays the same:

- lower `vw_cost` = better
- higher `vw_cost` = worse

## Why the cost transform matters

The notebook does not use raw reward directly. It converts reward into a bounded cost so VW can learn consistently.

The current design is:

- deterministic
- config-driven
- monotonic
- bounded to `[0, 1]`

That means the mapping is stable across runs and does not depend on fitting min/max values from the current dataset snapshot.

## Current output

Task 3 writes:

- `rewards/reward_labels_<USE_CASE_ID>_LOG_SCALED.parquet`
- `rewards/event_audit_<USE_CASE_ID>.parquet`

The reward-label file is used by Task 4.

## What happens next

After Task 3 finishes, the pipeline moves to:

1. `04_Task4_FINAL.ipynb`
   - joins reward labels to the decision log and action library

2. `5_build_vw_formatter.ipynb`
   - converts the joined data into VW multiline training blocks

3. `6_train_vw_model.ipynb`
   - trains VW on the train split only

4. `7_evaluate_vw_model.ipynb`
   - evaluates policy value on the holdout split

## What is correct about the design

- Campaign-sent rows are the right unit for VW bandit training.
- Raw historical events are used for reward attribution, not for direct training.
- Chronological splitting is the correct way to validate this type of model.
- VW expects `action:cost:probability`, so the current label contract is right.

## What to watch for

If almost all `vw_cost` values are close to `1.0`, the training signal is too compressed.

That does not mean the notebook is broken. It means the reward scale may need to be tuned so the model can see a clearer difference between good and bad outcomes.

## Quick summary

Task 3 is the reward-label stage. It turns campaign outcomes into VW-ready costs so the model can learn from them in Task 6 and be evaluated in Task 7.
