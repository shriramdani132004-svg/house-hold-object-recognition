# Phase 5 — Baseline Error Examples

Errors categorized at confidence >= 0.25 with greedy
same-class matching at IoU >= 0.5 over 1965
evaluated test images.
Each missed ground-truth object lands in exactly one primary
category; duplicate/spurious predictions count as false positives.

## Error categories

| Category | Count | Example image IDs |
|----------|-------|-------------------|
| low_confidence | 2856 | 000000000139, 000000000632, 000000001268, 000000001993, 000000002157 |
| poor_localization | 2183 | 000000000139, 000000000632, 000000000776, 000000001503, 000000002157 |
| missed_detection | 1001 | 000000000139, 000000000632, 000000001296, 000000001503, 000000002685 |
| false_positive | 901 | 000000000139, 000000000632, 000000001425, 000000003845, 000000004795 |
| cluttered_scene | 634 | 000000000139, 000000000632, 000000001503, 000000002157, 000000002685 |
| wrong_class | 119 | 000000001675, 000000002157, 000000002431, 000000003501, 000000007818 |

## Missed-object detail

- Ground-truth objects missed at the primary threshold: 5583
- Of those, small objects (box area < 1% of image): 3674 (65.8%)
- Test images with at least one categorized failure: 1587 / 1965

## Top class confusions (wrong class at IoU >= 0.5)

| Ground truth | Predicted | Count |
|--------------|-----------|-------|
| bottle | cup | 18 |
| cup | bottle | 10 |
| wine glass | cup | 9 |
| chair | couch | 8 |
| cup | bowl | 7 |
| bowl | cup | 7 |
| dining table | chair | 6 |
| couch | chair | 6 |
| cup | wine glass | 5 |
| cell phone | remote | 4 |
| book | laptop | 4 |
| keyboard | laptop | 3 |
| bottle | wine glass | 3 |
| bed | couch | 3 |
| bowl | dining table | 2 |

`cluttered_scene` is an image-level tag: images with 5+
ground-truth objects that contain at least one failure (it
overlaps the object-level categories above).
