# Phase 5 baseline sample review

Ground truth: thick box, prefix `GT`. Prediction: thin box, prefix `P`,
confidence shown for every detection at conf >= 0.25.
Selection is deterministic (seed from `configs/baseline.yaml`).

| # | File | Category | Image | Note | GT | Pred |
|---|------|----------|-------|------|----|------|
| 1 | 01_crowded_000000031296.jpg | crowded | 000000031296 | most ground-truth objects (38) | 38 | 12 |
| 2 | 02_single_object_000000000776.jpg | single_object | 000000000776 | exactly one ground-truth object | 1 | 0 |
| 3 | 03_small_object_000000522393.jpg | small_object | 000000522393 | smallest ground-truth box (area 0.000023) | 1 | 0 |
| 4 | 04_large_object_000000010583.jpg | large_object | 000000010583 | largest ground-truth box (area 1.000) | 1 | 2 |
| 5 | 05_class_chair_000000000139.jpg | class_chair | 000000000139 | contains 'chair' | 13 | 11 |
| 6 | 06_class_book_000000000632.jpg | class_book | 000000000632 | contains 'book' | 17 | 4 |
| 7 | 07_class_bottle_000000002685.jpg | class_bottle | 000000002685 | contains 'bottle' | 12 | 0 |
| 8 | 08_class_cup_000000002157.jpg | class_cup | 000000002157 | contains 'cup' | 12 | 10 |
| 9 | 09_class_dining_table_000000001993.jpg | class_dining_table | 000000001993 | contains 'dining table' | 4 | 3 |
| 10 | 10_class_bowl_000000001425.jpg | class_bowl | 000000001425 | contains 'bowl' | 1 | 2 |
| 11 | 11_class_potted_plant_000000013923.jpg | class_potted_plant | 000000013923 | contains 'potted plant' | 17 | 10 |
| 12 | 12_class_wine_glass_000000002431.jpg | class_wine_glass | 000000002431 | contains 'wine glass' | 4 | 5 |
| 13 | 13_class_cell_phone_000000001268.jpg | class_cell_phone | 000000001268 | contains 'cell phone' | 1 | 0 |
| 14 | 14_class_clock_000000001296.jpg | class_clock | 000000001296 | contains 'clock' | 2 | 1 |
| 15 | 15_random_000000021503.jpg | random | 000000021503 | seeded random pick (seed 42) | 3 | 2 |
| 16 | 16_random_000000355325.jpg | random | 000000355325 | seeded random pick (seed 42) | 6 | 6 |
| 17 | 17_random_000000507893.jpg | random | 000000507893 | seeded random pick (seed 42) | 1 | 1 |
| 18 | 18_random_000000513041.jpg | random | 000000513041 | seeded random pick (seed 42) | 6 | 5 |
| 19 | 19_random_000000507235.jpg | random | 000000507235 | seeded random pick (seed 42) | 3 | 3 |
| 20 | 20_random_000000323151.jpg | random | 000000323151 | seeded random pick (seed 42) | 4 | 4 |
