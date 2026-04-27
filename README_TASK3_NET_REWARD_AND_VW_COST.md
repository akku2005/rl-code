# Task 3: Net Reward and VW Cost

This document explains exactly how `3_build_reward_engine_FIXED.ipynb` calculates:

- `net_reward`
- `vw_cost`

It is written in plain English so the reward logic is easy to follow.

## Why this stage exists

VW does not train directly on raw business events.
It trains on a cost label in the form:

```text
action:cost:probability
```

So Task 3 takes campaign outcome data and converts it into a VW-ready cost.

## Step 1: Attribute events to a campaign send

Task 3 starts with:

- raw event history from `full_df.parquet`
- the decision log
- `reward_config.json`
- `funnel_config.json`

For each campaign send, it looks at the user’s events that happened:

- after the campaign send
- within the attribution window
- before the next campaign send

The notebook uses a 7-day attribution window by default.

## Step 2: Convert each rewarded event into a reward value

The notebook reads `reward_events` from `reward_config.json`.

For each event inside the attribution window:

- if the event has a fixed reward, that reward is used
- if the event uses `reward_column`, the notebook uses `event_value`
- if the event is not in `reward_events`, it is ignored

Example:

| Event name | Reward rule | Reward value |
|---|---|---:|
| `offer_viewed` | use `event_value` | `129905` |
| `lead_submitted` | fixed reward | `50` |
| `unknown_event` | not defined | ignored |

The notebook sums all valid event rewards for the campaign.

That sum is called:

```text
total_event_reward
```

## Step 3: Apply time decay

The notebook reduces the reward if the outcome happened later.

It calculates:

```text
time_decay_factor = time_decay_rate ^ attribution_delay_days
```

Then:

```text
decayed_reward = total_event_reward * time_decay_factor
```

### Plain-English meaning

- outcomes that happen sooner count more
- outcomes that happen later count less

If the event happens immediately, the decay factor is close to `1.0`.
If the event happens much later, the decay factor becomes smaller.

## Step 4: Subtract channel cost

Each channel has a business cost:

- WhatsApp: `0.39`
- SMS: `0.12`
- RCS: `0.25`
- Push: `0.03`
- Email: `0.05`

The notebook calculates:

```text
net_reward = decayed_reward - channel_cost
```

### Plain-English meaning

This says:

- good business outcome increases reward
- expensive channel usage reduces the final score

So `net_reward` is the business value after accounting for both outcome and delivery cost.

## Example: net reward

Suppose:

- `total_event_reward = 100`
- `time_decay_factor = 0.90`
- `channel_cost = 0.05`

Then:

```text
decayed_reward = 100 * 0.90 = 90
net_reward = 90 - 0.05 = 89.95
```

If no useful event happens:

- `total_event_reward = 0`
- `time_decay_factor = 1.0`
- `channel_cost = 0.39`

Then:

```text
net_reward = 0 - 0.39 = -0.39
```

That is why the raw `net_reward` can be negative.

## Step 5: Convert net reward into VW cost

VW minimizes cost, so the notebook converts `net_reward` into `vw_cost`.

The current notebook does this in a stable, config-driven way:

1. Choose a lower and upper reward bound
2. Clip `net_reward` into that range
3. Apply `log1p` compression
4. Invert the score so higher reward becomes lower cost
5. Clip the final result to `[0, 1]`

### The exact formula

```text
reward_floor = vw_cost_reward_floor
reward_ceiling = min(vw_cost_reward_ceiling, 5000.0)

bounded_reward = clip(net_reward, reward_floor, reward_ceiling)
reward_offset = bounded_reward - reward_floor
reward_span = reward_ceiling - reward_floor

scaled_reward = log1p(reward_offset) / log1p(reward_span)
vw_cost = 1 - scaled_reward
```

Then the notebook clips:

```text
vw_cost = clip(vw_cost, 0, 1)
```

## Plain-English meaning of the VW cost

- low `net_reward` becomes high `vw_cost`
- high `net_reward` becomes low `vw_cost`
- VW learns to prefer the actions with lower cost

## Example: VW cost

Imagine these values:

| Net reward | After transform | VW cost |
|---|---:|---:|
| `-0.39` | worst case | `1.00` |
| `-0.12` | low value | `0.97` |
| `50` | better value | around `0.70` |
| `5000` | capped best case | `0.00` |

The exact numbers depend on the configured reward floor and ceiling, but the direction is always the same:

- better reward = lower cost
- worse reward = higher cost

## What the notebook writes

Task 3 uploads:

- `rewards/reward_labels_<USE_CASE_ID>_LOG_SCALED.parquet`
- `rewards/event_audit_<USE_CASE_ID>.parquet`

The reward-label file is used by Task 4.

## Why this is acceptable for training

This design is valid for VW contextual bandits because VW expects a cost label, not a raw business reward.

The important part is that the transform is:

- deterministic
- monotonic
- bounded
- consistent across reruns

## Important note

The current cost distribution is still quite compressed near `1.0` for many rows.

That does not make the notebook wrong.
It means the learning signal may still be too weak, so the reward ceiling may need further tuning if you want stronger model separation.

## Short summary

Task 3 does this:

```text
raw events -> attributed rewards -> decayed reward -> channel cost -> net reward -> VW cost
```

That is the exact path used to prepare the label for VW training.
