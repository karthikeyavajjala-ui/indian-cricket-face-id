# Responsible AI statement

## Intended use

This project exists to **demonstrate and teach** how a facial-identification pipeline is built and
evaluated: detection → alignment → feature extraction → classification → decision policy → metrics.
It is intended for:

* learning and coursework in computer vision / machine learning;
* internal prototyping where participants have given informed consent;
* benchmarking pipeline choices on data you are allowed to process.

## Not intended for (and not safe for)

* **surveillance**, tracking, or monitoring of people in public or private spaces;
* **law enforcement**, forensic identification, or "finding" a suspect;
* **access control**, attendance, identity verification, or anti-fraud decisions;
* **any high-impact decision** about a person (employment, credit, insurance, education, benefits);
* inferring or claiming anything about **age, gender, ethnicity, emotion or health** (this build
  does none of this — and adding it would need its own review);
* identifying anyone **not** present in the training dataset: the model has no such capability and
  will answer `unknown`, which must not be reinterpreted as "this person is unknown to the world".

## Mandatory disclaimer (shown in the UI and returned by the API)

> This system is a demonstration of facial identification technology and should not be used for
> surveillance, law enforcement, access control, or high-impact decisions without appropriate
> safeguards. It recognises only the people present in its training dataset — it cannot identify
> every Indian cricketer.

## Safeguards implemented in the code

| safeguard | where |
|---|---|
| Refuses to answer below `confidence_threshold` (`status="unknown"`, `reason="low_confidence"`) | `ml/predictor.py::_decide` |
| Refuses when the top two classes are too close (`reason="ambiguous"`) | `ml/predictor.py::_decide` |
| Optional minimum prototype similarity for embedding mode | `PipelineConfig.min_similarity` |
| Never invents a prediction when no model exists (`status="model_missing"`) | `ml/predictor.py::identify` |
| No accuracy is reported when a held-out split is impossible (`accuracy: null`) | `ml/model.py::train_model` |
| Explicit caveat that augmentation leaks across splits and inflates metrics | `ml/model.py`, Model page |
| Rejections stored in the audit trail alongside matches | `predictions` table |
| No demographic/attribute inference anywhere in the pipeline | `ml/features.py` |
| Training endpoint protected by API key, one job at a time | `backend/routers/train.py` |
| Dataset is never auto-downloaded; you supply images you hold rights to | `scripts/prepare_dataset.py` |
| Bundled demo uses **AI-generated fictional people** only | `dataset_demo/`, `scripts/make_demo_dataset.py` |

## Known technical limitations (be explicit with users)

1. **Small-data overfitting.** With 6–30 images per player the model memorises photographic
   conditions rather than robust identity features. Accuracy on new lighting/angles/ages drops.
2. **Augmentation leakage.** Variants of one photo can appear on both sides of the train/validation
   split, so reported metrics are optimistic; only fresh photographs give an honest estimate.
3. **Appearance drift.** Beard, hair, weight, eyewear, helmets/caps, celebrating poses and injury
   can all push a correct person below threshold.
4. **Class imbalance.** Players with fewer images get lower per-player recall — check
   `per_class` before trusting the macro numbers.
5. **Demographic bias.** Any face dataset under-represents some groups; this pipeline inherits
   whatever bias is in the images you supply and has **not** been fairness-tested.
6. **Detector fragility.** Haar misses profile faces and heavy occlusion; even YuNet degrades at
   extreme pose, low light, motion blur or low resolution.
7. **No liveness / anti-spoofing.** A photo of a photo (or a screen) will be treated as a face.
8. **Image → identity is reversible in the sense that the system is a biometric matcher.** Treat
   the trained model and the `predictions` table as sensitive biometric data.

## Privacy and the law

Face images and face templates are **sensitive personal data**.

* In India, processing them engages the **Digital Personal Data Protection Act, 2023** — you need a
  lawful basis, purpose limitation, notice/consent where applicable, security safeguards, and
  deletion when the purpose ends. Other jurisdictions add more (GDPR Art. 9 for EU/EEA, BIPA in
  Illinois, and various state biometric laws in the US).
* Do **not** process images of minors, bystanders in the background, or people who have not
  consented, for anything beyond a private demo.
* This repo ships no real-person imagery. If you add real photographs, that is your processing
  activity and your liability — including personality/publicity rights and copyright in each file.

### Retention controls built in

* `IC_STORE_UPLOADS=0` — do not persist uploaded frames or face crops.
* `save=false` per request — skip persistence for a single call.
* `DELETE /api/predictions` — clear the audit trail.
* `make clean` / delete `data/` — remove DB, uploads and caches.

### Recommended before any real deployment

1. Written lawful basis + consent/notice design, and a DPIA/impact assessment.
2. Authentication and authorisation on every read endpoint (this demo has none).
3. Retention schedule, encryption at rest, access logging, and a deletion workflow.
4. Independent evaluation on a held-out, consented, demographically representative set, with
   per-group error rates published.
5. Human-in-the-loop review for any consequential outcome — never automated decisions.
6. A published model card and an incident/opt-out process.

## What this project deliberately does not do

* No scraping, no bundling of copyrighted player photographs, no auto-download of datasets.
* No "identify any Indian cricketer" claim — the catalog is metadata; recognition is limited to
  whatever you trained on.
* No hidden guessing: when the model is unsure, the answer is **“Unknown / Low Confidence”** with
  the runner-up scores visible for scrutiny.
