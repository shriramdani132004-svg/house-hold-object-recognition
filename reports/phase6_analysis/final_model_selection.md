# Phase 6 — Final Model Selection

Selected method: **REPLAY**

Rationale: Replay is at least as good as naive on every primary measured criterion and strictly better on at least one; select replay.

Measured detail: final_accuracy: naive 0.0235 vs replay 0.0538 (replay better); final_forgetting: naive 0.6209 vs replay 0.5364 (replay better); old_knowledge_retention: naive 0.0235 vs replay 0.0538 (replay better); average_incremental_accuracy: naive 0.0458 vs replay 0.0632 (replay better).

## Primary measured criteria

| Criterion | Naive | Replay | Verdict |
|---|---|---|---|
| final_accuracy | 0.0235 | 0.0538 | replay better |
| final_forgetting | 0.6209 | 0.5364 | replay better |
| old_knowledge_retention | 0.0235 | 0.0538 | replay better |
| average_incremental_accuracy | 0.0458 | 0.0632 | replay better |

## Full comparison

| Metric | Naive | Replay |
|---|---|---|
| Final accuracy | 0.0235 | 0.0538 |
| Final mean forgetting | 0.6209 | 0.5364 |
| Old-class retention (final) | 0.0235 | 0.0538 |
| Average incremental accuracy | 0.0458 | 0.0632 |
| Training time (s) | 2258.2400 | 5387.2150 |

## Tradeoffs

- Both methods share the identical architecture (SmallConvNet (width 32, 64x64 input)), so final model size and inference cost are the same; the replay checkpoint file is larger only because its payload carries the bounded replay memory.
- Replay training took longer per experience (replay batch loading); training time is not a quality criterion for the frozen inference model.

## Source experiment

- CORe50 / NIC / inc / run 0, 79 experiences, seed 42
- artifacts: reports/phase5_nic/
