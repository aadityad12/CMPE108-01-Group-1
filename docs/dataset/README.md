# Dataset log

Images and manifests live in `data/` and are not committed. This file records
how they were made so anyone can rebuild them.

## Writers

| Writer | Source | Role | Distinct formulas | Test | Support budgets |
|---|---|---|---|---|---|
| team01 (Aaditya) | Handwritten for this project | Evaluation | 159 | 50 | 10, 25, 50, 100 |
| CROHME23:J021–J025 | CROHME 2023 training set | Evaluation | 61–71 each | 25 | 10, 25 |
| CROHME23:J011, J012, J015 | CROHME 2023 training set | Tuning only | 35–36 each | 25 | 10 |

- The CROHME 2023 writers were collected after the CoMER checkpoint was
  trained, so the base model has not seen them.
- The three tuning writers are only used to pick learning rates and step
  counts, never for reported results. CROHME 2019 User002 (198 formulas) was
  tried first, but the base model already scores 100% on it (it was in
  CoMER's training data), so it shows nothing about tuning.
- Splits are frozen with seed 42 and checked with
  `wlm.leakage --require-formula-disjoint`: no formula is in both a writer's
  support pool and test set.

## team01 collection (2026-11-17)

- 159 prompts from `prompts/formulas.txt`, copied by hand in order, one per line.
- Blue/black pen on grid paper, 16 phone photos (one per page), indoor light,
  one session. Some pages have shadows or carpet at the edges.
- Cut into one image per formula with:

```bash
python scripts/crop_pages.py --photos data/raw/team/aaditya --writer team01 \
  --shadow-pages 2,6,8,10,16 --corrections docs/dataset/team01-corrections.tsv
```

- `--shadow-pages` lists pages where part of the paper is shaded; they use a
  lower brightness cutoff so the shaded paper is not mistaken for background.
- A crossed-out line on page 11 is removed automatically (dense scribble).
- Every crop was checked by eye against its label. 11 labels were changed to
  match what was written (slashed fractions, a dot before `dx`, dot height,
  one sum limit). See [team01-corrections.tsv](team01-corrections.tsv).
- Crops are saved as dark ink on pure white (grid and shadows removed), like
  the training images. Raw grayscale crops gave the base model 0% exact on
  the 109 non-test images; clean crops give 51.4%. Scale 1.0 was best
  (1.5: 43.1%, 2: 44.0%, 3: 17.4%), chosen on non-test images only.
- Audit: 159 rows, no out-of-vocabulary tokens.
- Consent: the writer agreed to use this handwriting in the report and demo.

## Public writers

```bash
PYTHONPATH=. python scripts/import_public_writers.py
```

Labels converted from raw LaTeX to the model's token format; 5 of 349 CROHME
2023 labels used symbols outside the vocabulary and were skipped. Scans are
cropped to the ink, contrast-stretched and scaled by 0.6 (scale picked with the
base model on 60 images from other 2023 writers).

## Splits

```bash
python -m wlm.prepare writer-splits --manifest data/team.jsonl --output data/writers/team --seed 42
python -m wlm.prepare writer-splits --manifest data/public.jsonl \
  --writers CROHME23:J011,CROHME23:J012,CROHME23:J015 \
  --output data/writers/dev --test-size 25 --budgets 10 --seed 42
python -m wlm.prepare writer-splits --manifest data/public.jsonl \
  --writers CROHME23:J021,CROHME23:J022,CROHME23:J023,CROHME23:J024,CROHME23:J025 \
  --output data/writers/public --test-size 25 --budgets 10,25 --seed 42
```
