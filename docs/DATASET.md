# Building your dataset

The trainer reads **folders of images**. It never downloads player photographs, and the project
ships none of real people. You supply images you are legally allowed to process.

```
dataset/
├── Virat_Kohli/
│   ├── 01_face.jpg
│   ├── 02_face.jpg
│   └── …                      ← 10–30 images per player is the sweet spot
├── Rohit_Sharma/
├── Jasprit_Bumrah/
└── …                          ← at least 2 folders, any number above that
```

Rules:

* Folder name = class label. `Virat_Kohli` displays as **Virat Kohli**.
* Folders starting with `.` or `_` are ignored (handy for hold-outs, e.g. `_holdout/`).
* Supported formats: `.jpg .jpeg .png .webp .bmp .jfif`.
* Nested sub-folders are allowed and scanned recursively.
* File names do not matter; duplicates by content do (they inflate metrics without adding signal).

## 1. Where to legally get images

| source | notes |
|---|---|
| your own photographs | best case; you hold the rights. Never photograph people without consent for this purpose. |
| CC-licensed media (Wikimedia Commons, Flickr CC, Openverse) | check the specific licence (attribution, share-alike, no-derivatives) and keep the attribution record |
| press/agency photos, broadcast stills, social-media screenshots | **do not use** — almost always copyright-restricted and often personality-rights restricted |
| scraped search-engine results | **do not use** — provenance and licence are unknown |

Keep a `LICENCES.md` in each folder recording the source URL, licence and attribution for every
file. `scripts/prepare_dataset.py --audit` reminds you that this is on you.

## 2. What makes a good image set

* Faces at 150 px or larger, front-facing, evenly lit, not motion-blurred.
* Vary: indoor/outdoor, daylight/floodlights, expressions (neutral, smiling, celebrating),
  head angle within ±30°, distance, camera and compression quality, and — if you have them —
  different ages and facial hair states.
* Prefer 1 face per image; if there are several, the **largest** face is used.
* Crop close enough that the face is a meaningful share of the frame (group photos of a whole
  team give tiny faces and weak descriptors).
* Avoid: heavy filters, watermarks over the face, extreme low-key lighting, sunglasses/helmet
  covering the eyes, images where the face is <40 px.

Target: **10–30 images per player**. Below ~6 the model overfits and the metrics become noise; the
trainer prints a warning per player when images are scarce.

## 3. Tooling

```bash
# create the tree, print the roster and the exact folder names to use
python scripts/prepare_dataset.py --init
python scripts/prepare_dataset.py --roster

# validate: readability, size, blur, face detectability, multi-face, duplicates
python scripts/prepare_dataset.py --audit --min-images 8 --json-out audit.json

# optional: resize to max 1600 px, re-encode to JPEG q92, strip EXIF (privacy + speed)
python scripts/prepare_dataset.py --optimize
```

The audit never deletes anything - it reports, and you decide.

## 4. Augmentation and honest metrics

`ml/pipeline.py` derives `IC_AUGMENT_PER_IMAGE` (default 6) variants per image: small rotation,
scale, translation, brightness/contrast, colour temperature, blur and noise. This helps a small
dataset generalise and makes the classifier robust to broadcast-style variation.

**Important caveat.** Because variants are derived from the same source photo, a variant can land
in the validation split while its sibling is in training. That inflates held-out accuracy compared
with genuinely unseen photographs. The Model page states this explicitly, cross-validation is
reported separately (and is *also* optimistic), and no accuracy is reported at all when a split is
impossible. Evaluate on **new, untouched photographs** before believing any number.

To get an honest number:

1. train as usual, note the metrics;
2. add 3–5 fresh photos per player that were never used for training;
3. either hold them out via a `dataset/_holdout/<Player>/` folder and score them manually, or
   temporarily move them out of `dataset/`, retrain, and run
   `python -m ml.predict --dir _holdout/Virat_Kohli`.

## 5. Enlarging the roster

`backend/catalog.py` holds profile facts for 12 Indian players (role, batting/bowling style,
jersey number, career highlights). To support a player who is not in the catalog:

```bash
mkdir -p dataset/Some_Player && cp your_images/*.jpg dataset/Some_Player/
python -m ml.train --dataset dataset
```

A dataset folder without a catalog entry still works: the API creates a stub profile that says the
catalog has no verified facts for it, and the UI offers to add them. Adding a real entry to
`backend/catalog.py` gives you the full profile card (only add facts you can verify).

## 6. Housekeeping

* Keep `dataset/` out of version control (`.gitignore` already excludes it) — face images are
  personal data.
* Delete the images, `data/uploads/`, the `predictions` table and `data/app.db` when you are done.
* If you publish a model trained on real faces, you are publishing a biometric system: think about
  consent, retention and who can query it.
