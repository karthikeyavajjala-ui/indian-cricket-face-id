# dataset/ — put your own images here

One folder per supported player; the folder name is the class label
(`Virat_Kohli` is displayed as **Virat Kohli**).

```
dataset/
├── Virat_Kohli/        01.jpg 02.jpg …
├── Rohit_Sharma/       01.jpg …
├── Jasprit_Bumrah/     …
└── …                   (≥ 2 players with images; 10–30 images each is ideal)
```

The app never downloads player photographs — you supply images you are legally allowed to use
(your own photos, CC-licensed media with attribution, or material you have explicit permission to
process). Do not scrape news agencies, broadcasters or social platforms.

**You are responsible for the licence and the personality/publicity rights of every file here.
Face images are sensitive personal data — keep this folder out of version control and delete it
when you are done.**

Supported formats: `.jpg .jpeg .png .webp .bmp .jfif`
Folders starting with `_` or `.` are ignored (handy for hold-out sets, e.g. `_holdout/`).

## Commands

```bash
python scripts/prepare_dataset.py --roster          # folder names from the player catalog
python scripts/prepare_dataset.py --audit --min-images 8
python scripts/prepare_dataset.py --optimize        # resize to 1600px, strip EXIF
python -m ml.train --dataset dataset                # train + evaluate -> models/model.pkl
```

Read `docs/DATASET.md` for guidance on sourcing, image quality, augmentation leakage and how to
measure accuracy honestly.
