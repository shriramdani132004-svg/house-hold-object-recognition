# Dataset

## Selected Dataset

**COCO — Common Objects in Context (COCO 2017 release)**

The dataset for this project is the COCO 2017 object-detection release:
the `train2017` and `val2017` image splits plus the official
`annotations_trainval2017` annotation archive (which contains the
annotations for both splits).

## Official Source

- Website: <https://cocodataset.org/>
- Official download page: <https://cocodataset.org/#download>
- Data host: the public `images.cocodataset.org` bucket (served over the
  standard AWS S3 REST endpoint `https://s3.amazonaws.com/images.cocodataset.org`
  for this project — see *Acquisition Method*)
- Official terms of use: <https://cocodataset.org/#termsofuse>

## Dataset Version

- COCO 2017 detection release (images released as `train2017` / `val2017` /
  `test2017`).
- Annotation archive: `annotations_trainval2017.zip`
  (server `Last-Modified: 2018-07-10`).
- Image archives used: `train2017.zip` and `val2017.zip`
  (both server `Last-Modified: 2018-07-11`).

## License

Taken verbatim in substance from the official COCO Terms of Use
(<https://cocodataset.org/#termsofuse>):

- **Annotations and website:** licensed by the COCO Consortium under a
  **Creative Commons Attribution 4.0 License (CC BY 4.0)** —
  <https://creativecommons.org/licenses/by/4.0/>. Commercial and
  portfolio use is permitted with attribution.
- **Images:** the COCO Consortium does **not** own the copyright of the
  images. Use of the images must abide by the **Flickr Terms of Use**;
  users accept full responsibility for the use of the dataset. Individual
  images carry their own (mostly Creative Commons / Flickr) licenses.
  Practical consequence: attribution is mandatory, and raw images should
  not be redistributed as if they were ours — demos/reports should link
  back to COCO as the source.
- Downloading the dataset constitutes agreement to the official Terms of
  Use.

## Why This Dataset Was Selected

Compared against the other candidates (table below), COCO 2017 scores
best on the project's selection criteria:

1. **Household-object relevance:** ~40 of its 80 categories are everyday
   household objects (cup, bottle, bowl, fork, knife, spoon, chair, couch,
   bed, dining table, tv, laptop, remote, microwave, oven, sink,
   refrigerator, book, clock, vase, toothbrush, …) — far denser in
   household classes than PASCAL VOC.
2. **Annotation quality:** professionally drawn bounding boxes (plus
   segmentation polygons) in a single, well-documented JSON format.
3. **Licensing clarity:** explicit official Terms of Use with CC BY 4.0
   annotations.
4. **Accessibility:** plain HTTP archive downloads from the official
   host, no login, no API key, no scraper.
5. **Diversity:** 118K train / 5K val real-world scenes (COCO: 330K
   images overall, 80 categories per the official site).
6. **Reasonable size:** official splits are downloadable individually;
   the complete detection data is ≈19 GiB of archives (see
   *Dataset Size*) and fits comfortably on disk.
7. **Ease of automated acquisition:** single zip files with published
   byte sizes; resumable range requests.
8. **YOLO compatibility:** COCO format is natively supported by the
   Ultralytics YOLO toolchain (`coco.yaml` and auto-conversion), which
   is the planned training pipeline.

## Dataset Size

**Complete COCO 2017 detection data (official figures from the download
page, verified via HTTP HEAD against the official host):**

| File | Contents | Bytes |
|---|---|---|
| `zips/train2017.zip` | 118K training images | 19,336,861,798 (~18.01 GiB) |
| `zips/val2017.zip` | 5K validation images | 815,585,330 (~0.76 GiB) |
| `annotations/annotations_trainval2017.zip` | annotations for train **and** val | 252,907,541 (~241 MiB) |

**What this project acquires (the documented acquisition
specification for Phase 2 — all three archives):**

- `annotations_trainval2017.zip` — **acquired** (full annotations for
  both splits: instances, captions, person keypoints).
- `val2017.zip` — **acquired** (the complete official 5,000-image
  validation split).
- `train2017.zip` — **acquired** (the complete official 118,287-image
  training split; 18.01 GiB, fetched in resumable chunks — throughput
  on the official endpoint from this machine measured between ≈0.4 and
  ≈5.5 MB/s across sessions).

**Verified after acquisition** (`scripts/verify_dataset.py`, PASS):

- Archives: 20,405,354,669 bytes total (all three files byte-exact vs
  official host; annotations MD5 matches official ETag; full ZIP CRC
  passed for every archive).
- Extracted: annotations 834,416,290 bytes + 118,287 train JPEGs
  19,314,466,396 bytes + 5,000 val JPEGs 814,705,164 bytes →
  **41,368,942,519 bytes (~38.53 GiB)** total under `data/raw/coco/`.
- Image counts on disk: 118,287 train and 5,000 val JPEGs (each
  exactly matches the corresponding annotation file).
- Annotation content: 118,287 train images / 860,001 train instances,
  5,000 val images / 36,781 val instances, 80 categories (identical
  across splits).

## Relevant Classes

COCO detection has **80 categories** (the project's final class subset
is **not** decided here — that belongs to Phase 3). The
household-relevant categories available in the dataset are:

`bottle`, `wine glass`, `cup`, `fork`, `knife`, `spoon`, `bowl`,
`banana`, `apple`, `sandwich`, `orange`, `broccoli`, `carrot`,
`hot dog`, `pizza`, `donut`, `cake`, `chair`, `couch`, `potted plant`,
`bed`, `dining table`, `tv`, `laptop`, `mouse`, `remote`, `keyboard`,
`cell phone`, `microwave`, `oven`, `toaster`, `sink`, `refrigerator`,
`book`, `clock`, `vase`, `scissors`, `teddy bear`, `toothbrush`,
`hair drier`.

Food classes (`banana` … `cake`) and personal items (`scissors`,
`toothbrush`, `teddy bear`) are included in this candidate list because
they appear in everyday home scenes; Phase 3 will finalize the exact
subset based on per-class instance counts (measured from the acquired
annotation files).

## Annotation Format

Official **COCO JSON** format (one JSON file per annotation type and
split):

- `instances_train2017.json`, `instances_val2017.json` — object
  detection + segmentation. Structure: top-level `images`, `annotations`,
  `categories` lists. Each annotation has `image_id`, `category_id`,
  `bbox = [x, y, width, height]` in absolute pixels (top-left origin),
  `area`, `iscrowd`, and `segmentation` (polygon list or compressed RLE
  mask).
- `captions_train2017.json`, `captions_val2017.json` — image captions
  (not needed for this project).
- `person_keypoints_train2017.json`, `person_keypoints_val2017.json` —
  person keypoints (not needed for this project).
- `categories`: 80 detection classes with non-contiguous COCO category
  IDs (gaps exist between IDs).

Conversion to YOLO format is **Phase 3 work** and is not performed in
Phase 2.

## Acquisition Method

- Script: `scripts/download_dataset.py` (run from the project root).
- Sources: the official COCO archives listed above, downloaded with
  resumable HTTP range requests, automatic retry with backoff, progress
  output, exact-size verification, MD5 verification where the host
  provides a plain-MD5 ETag, and ZIP CRC integrity checks before
  extraction.
- Endpoint note: the official vanity host `images.cocodataset.org` has
  no valid TLS certificate for its hostname, and its plain-HTTP body
  transfers are reset on this network. The identical official bucket is
  therefore fetched through the standard S3 REST endpoint
  `https://s3.amazonaws.com/images.cocodataset.org/...`, with the
  original `http://images.cocodataset.org/...` host as a fallback. Same
  bucket, same objects, same byte sizes (verified by HEAD).
- Layout produced (raw, never modified afterwards):

  ```
  data/raw/coco/
  ├── zips/annotations_trainval2017.zip
  ├── zips/train2017.zip
  ├── zips/val2017.zip
  ├── annotations/instances_train2017.json
  ├── annotations/instances_val2017.json
  ├── annotations/captions_*.json
  ├── annotations/person_keypoints_*.json
  ├── train2017/000000xxxx.jpg   (118,287 images)
  └── val2017/000000xxxx.jpg     (5,000 images)
  ```

- Verification utility: `scripts/verify_dataset.py`.
- Raw data and archives are excluded from Git by `.gitignore`; only the
  acquisition/verification scripts and metadata files are committed.

## Citation / Attribution

Required attribution for use of COCO (dataset paper; metadata verified
via Crossref for DOI `10.1007/978-3-319-10602-1_48`):

```bibtex
@inproceedings{lin2014microsoft,
  title     = {Microsoft COCO: Common Objects in Context},
  author    = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and
               Hays, James and Perona, Pietro and Ramanan, Deva and
               Doll{\'a}r, Piotr and Zitnick, C. Lawrence},
  booktitle = {European Conference on Computer Vision (ECCV)},
  pages     = {740--755},
  publisher = {Springer},
  year      = {2014},
  doi       = {10.1007/978-3-319-10602-1_48}
}
```

When displaying or redistributing results, credit both the COCO
Consortium (annotations, CC BY 4.0) and the original image sources
(Flickr).

## Limitations

- **Image licensing is per-image:** COCO images remain under their
  original Flickr licenses; the official Terms of Use shift
  responsibility to the user. Fine for research/portfolio use with
  attribution; do not redistribute raw images as our own.
- **Large footprint:** the complete acquired dataset occupies ≈38.53
  GiB on disk (archives + extracted images). It is git-ignored, so
  losing `data/raw/coco/` means re-downloading ≈19 GiB.
- **No official published checksums:** COCO publishes byte sizes and
  ETags, not MD5/SHA manifests. Verification therefore uses exact
  byte-size matching, the S3 ETag MD5 where available, and full ZIP CRC
  testing.
- **Long-tailed class distribution:** many COCO classes are rare;
  per-class counts must be analyzed in Phase 4 and the household class
  subset fixed in Phase 3.
- **Not purely household scenes:** COCO contains many outdoor/person/
  vehicle images; the project will filter to household-relevant classes
  (and possibly scenes) in Phase 3.
- **No test-set labels:** `test2017` is unlabeled (evaluation server
  only), so all splits used here are train/val-derived.

---

## Candidates Considered (research record)

| Criterion | **COCO 2017** (selected) | Open Images V7 | PASCAL VOC 2007/2012 |
|---|---|---|---|
| Official source | <https://cocodataset.org/> (download page + Terms of Use verified) | <https://storage.googleapis.com/openimages/web/index.html> (description + download pages verified) | <http://host.robots.ox.ac.uk/pascal/VOC/> (homepage verified) |
| Images | 118K train / 5K val / 40K test (2017 detection splits) | ~9M total; 1.9M with boxes; 1.7M train images | ~9,963 (2007), ~11.5K (2012) |
| Annotations | Boxes (+polygons/masks/keypoints), COCO JSON | 16M boxes / 600 boxable classes, CSV files | Bounding boxes, Pascal XML; 20 classes |
| Household relevance | **High** — ~40/80 classes are household objects | High — 600 classes incl. many household items | **Low** — only ~6 household classes (bottle, chair, diningtable, pottedplant, sofa, tvmonitor) |
| License (official) | Annotations: **CC BY 4.0** (COCO Consortium); images: Flickr Terms of Use | Annotations: **CC BY 4.0** (Google LLC); images: CC BY 2.0 as listed, with official disclaimer to verify each image | Homepage reviewed does not state a clear annotation license; images are from Flickr with their own licenses |
| Download method | Official HTTP zips / gsutil (no auth) | Manual per-image downloads, TFDS, or FiftyOne (full dataset ≈ **0.5 TiB** per official page) | Official tar files from Oxford servers (no auth) |
| Approx. size | 19 GiB of archives; **fully acquired (all three archives)** | ~0.5 TiB full; class subsets via tooling | ~0.87 GiB (2007), ~3.59 GiB (2012) per TFDS catalog of official files |
| Portfolio/commercial notes | Annotations free with attribution; images individually licensed | Annotations free with attribution; image-license disclaimer | Research-site origins; licensing less explicit |
| YOLO pipeline fit | **Native** (`coco.yaml`, auto-conversion) | Needs custom CSV→YOLO conversion + image fetch tooling | Needs VOC→YOLO conversion |
| Documentation quality | Very good (official site, tools, API, community) | Very good (official site, papers, tooling) | Good (official site, devkit) |

Other datasets considered but excluded: **Objects365** (official site
`objects365.org` failed TLS validation during this research, so licensing
could not be verified from a primary source; full dataset also very
large) and **LVIS** (excellent fine-grained annotations but built for a
more complex evaluation protocol; its images overlap COCO anyway).
