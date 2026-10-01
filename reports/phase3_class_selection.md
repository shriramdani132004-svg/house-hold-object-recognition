# Phase 3 — Final Class Selection

Generated from measured COCO 2017 statistics
(`reports/phase3_category_analysis.md`, produced by
`python scripts/analyze_categories.py`). No statistics below are
hand-entered.

## Selection criteria

Applied to the 34-category candidate pool from `configs/dataset.yaml`:

1. **Representation floor:** >= 2,000 COCO train instances and
   >= 1,000 COCO train images (`selection.min_instances_per_class` /
   `selection.min_images_per_class` in `configs/dataset.yaml`).
2. **Household relevance:** in-home objects only; no person, animal,
   vehicle, food, or outdoor-only classes (the pool contains none of
   these anyway).
3. **Live-camera demo value:** classes a webcam demo can realistically
   show (desk, living room, kitchen surfaces).
4. **Visual learnability:** distinct, reasonably sized objects; avoid
   mutually confusable or very thin/small objects.
5. **Class balance:** avoid adding classes that would deepen the
   long tail without adding scene diversity.

## Final classes — 19 selected

Order below is the final YOLO class ID order (`configs/classes.yaml`),
sorted by train instance count.

| ID | Class | COCO id | Train instances | Train images | % train images | Val instances |
|---:|---|---:|---:|---:|---:|---:|
| 0 | chair | 62 | 38,491 | 12,774 | 10.8% | 1,791 |
| 1 | book | 84 | 24,715 | 5,332 | 4.5% | 1,161 |
| 2 | bottle | 44 | 24,342 | 8,501 | 7.2% | 1,025 |
| 3 | cup | 47 | 20,650 | 9,189 | 7.8% | 899 |
| 4 | dining table | 67 | 15,714 | 11,837 | 10.0% | 697 |
| 5 | bowl | 51 | 14,358 | 7,111 | 6.0% | 626 |
| 6 | potted plant | 64 | 8,652 | 4,452 | 3.8% | 343 |
| 7 | wine glass | 46 | 7,913 | 2,533 | 2.1% | 343 |
| 8 | cell phone | 77 | 6,434 | 4,803 | 4.1% | 262 |
| 9 | clock | 85 | 6,334 | 4,659 | 3.9% | 267 |
| 10 | tv | 72 | 5,805 | 4,561 | 3.9% | 288 |
| 11 | couch | 63 | 5,779 | 4,423 | 3.7% | 261 |
| 12 | remote | 75 | 5,703 | 3,076 | 2.6% | 283 |
| 13 | sink | 81 | 5,610 | 4,678 | 4.0% | 225 |
| 14 | laptop | 73 | 4,970 | 3,524 | 3.0% | 231 |
| 15 | bed | 65 | 4,192 | 3,682 | 3.1% | 163 |
| 16 | keyboard | 76 | 2,855 | 2,115 | 1.8% | 153 |
| 17 | refrigerator | 82 | 2,637 | 2,360 | 2.0% | 126 |
| 18 | mouse | 74 | 2,262 | 1,876 | 1.6% | 106 |

Scene coverage: living room (chair, couch, tv, remote, clock,
potted plant), dining/kitchen (dining table, bottle, cup, bowl, wine
glass, sink, refrigerator), study/desk (book, laptop, keyboard, mouse,
cell phone), bedroom (bed).

## Excluded candidates — 15

Passed the representation floor but excluded on usefulness/learnability:

| Class | Train instances | Train images | Reason for exclusion |
|---|---:|---:|---|
| handbag | 12,354 | 6,841 | Bag trio (handbag/backpack/suitcase) is mutually confusable and travel/outdoor-oriented; class budget spent on in-home demo classes |
| backpack | 8,720 | 5,528 | Same bag-trio confusion; mostly outdoor/commute scenes |
| knife | 7,770 | 4,326 | Thin cutlery: poor live-camera visibility, confusable with fork/spoon |
| vase | 6,613 | 3,593 | Tabletop decor of similar scale/silhouette to bottle and wine glass; potted plant already covers decor |
| suitcase | 6,192 | 2,402 | Bag trio; travel-only use case |
| spoon | 6,165 | 3,529 | Thin cutlery; mutual confusion with fork/knife |
| fork | 5,479 | 3,555 | Thin cutlery; mutual confusion with knife/spoon |
| teddy bear | 4,793 | 2,140 | Well represented, but single novelty object; excluded to keep the set near 19 with better scene balance |
| toilet | 4,157 | 3,353 | Household, but bathroom scenes are outside the intended live-demo scenarios |
| oven | 3,334 | 2,877 | COCO oven boxes overlap kitchen cabinetry/stove boundaries (ambiguous extent); kitchen covered by sink + refrigerator + tableware |

Below the representation floor:

| Class | Train instances | Train images | Reason for exclusion |
|---|---:|---:|---|
| toothbrush | 1,954 | 1,007 | Under 2,000-instance floor; very small object, weak demo value |
| microwave | 1,673 | 1,547 | Under floor; appliance box also confusable with oven/cabinet |
| scissors | 1,481 | 947 | Under floor on both instances and images |
| toaster | 225 | 217 | Extremely low representation |
| hair drier | 198 | 189 | Extremely low representation |

## Class-balance considerations

- Selected classes span 2,262 (mouse) to 38,491 (chair) train
  instances — a ~17:1 max/min ratio, characteristic of COCO's long tail
  and far better balanced than including the sub-2,000 tail classes.
- The most frequent class (chair) holds 38,491 of the ~207,416
  selected-class train annotations (~19%, before crowd exclusion), so
  no single class dominates the loss.
- Thresholds were set after measuring the candidate distribution
  (`configs/dataset.yaml`): a 3,000-instance floor would have dropped
  keyboard, mouse, and refrigerator — the highest-value desk/kitchen
  demo classes — so 2,000 was used instead.
- Per-split class distributions and imbalance indicators are reported
  in `reports/phase3_dataset_statistics.md` after preparation.
