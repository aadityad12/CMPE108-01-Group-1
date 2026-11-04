# Team plan: who does what

Proposed 2026-11-03 from `GUIDE.html` section 15. Edit freely; this is the one place
the split of work is written down.

| Person | Owns | Detailed plan |
|---|---|---|
| Jeriel | Model setup, training, **all SJSU HPC jobs** | [plan-hpc.md](plan-hpc.md) |
| Aidana | Team handwriting data, labels, writer metadata, frozen splits, dataset audit | below |
| Aaditya | Evaluation, error analysis, demo, report assembly | [plan-evaluation-and-demo.md](plan-evaluation-and-demo.md) |

Everyone reviews someone else's labels, and everyone should be able to explain the
split protocol, the trainable parameters of each method, and one recognition failure.

## Status on 2026-11-03

Done:

- Pinned CoMER source, converted base checkpoint (hash reproduced on two clean setups),
  9 passing tests, Mac CPU environment lock
- Base reference on CROHME 2019: 62.64% exact token match (1,199 images, CPU)
- Error-analysis tool (`wlm.compare`) and a CPU pilot of all three adaptation methods
  (see [eval-log.md](eval-log.md))

Not started:

- **Team handwriting data.** Nothing else in the writer study can run without it
- Any HPC job
- Development-writer tuning

## Data tasks (Aidana)

This is the critical path. The protocol needs **150 distinct formulas per evaluation
writer** (100 support pool + 50 test), so about 450 labeled crops for three writers.

- [ ] Prompt list of ~200 formulas checked against
      `third_party/CoMER/comer/datamodule/dictionary.txt` (fractions, scripts, radicals,
      sums, mixed lengths). Same prompts for every writer
- [ ] 10-expression pilot per writer → `data/team.jsonl` (format: `examples/team_sample.jsonl`)
      → `python -m wlm.prepare audit` → hand to Jeriel for the H1/H2 overfit check
- [ ] Full collection in at least two capture sessions; record phone, lighting, session
- [ ] Second person verifies every label against the photo; keep a correction log
- [ ] `python -m wlm.prepare writer-splits --manifest data/team.jsonl --output data/writers --seed 42`,
      then **freeze it** (share the folder; never regenerate)
- [ ] Development writers for tuning (a fourth person, or public writers), kept separate
      from the three evaluation writers
- [ ] Dataset log: counts, sessions, exclusions, consent to use handwriting in report/demo

Fallback if 150 per writer is not reachable in time: 100 per writer (50 support + 50 test)
with budgets 10/25/50. That needs a small change to `writer-splits` (it currently
hard-codes 10/25/50/100) and must be decided **before** the split is frozen.

## Order of work

| Step | Who | Depends on |
|---|---|---|
| Pilot data (10 per writer) | Aidana | — |
| HPC environment + GPU timing pilot (H0, H1) | Jeriel | — |
| Error-analysis tool, report draft, demo dry run | Aaditya | — |
| Full collection + split freeze | Aidana | pilot audited |
| Development tuning (H2) | Jeriel | development-writer data |
| Base per writer + sweep seed 0 (H3, H4) | Jeriel | frozen splits, tuned settings |
| Analysis, figures, error review | Aaditya | seed 0 results |
| Seeds 1–2, final tables | Jeriel → Aaditya | seed 0 checked |
| Report + presentation | all | results |
