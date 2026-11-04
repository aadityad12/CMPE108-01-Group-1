# Write Like Me: Few-Shot Writer Personalization of a Handwritten Math Recognizer

**CMPE 108, Section 01, Group 1:** Jeriel Zhi Hao Goh, Aidana Muralieva, Aaditya Desai
**Repository:** <https://github.com/aadityad12/CMPE108-01-Group-1> (instructor and TA invited)
**Status:** draft for Project Check-in 3, 2026-11-03. Sections marked *(planned)* are
outlines; every number in this draft was measured and is logged in `docs/eval-log.md`.

---

## Abstract (tentative)

Handwritten mathematical expression recognition (HMER) models are trained on pooled data
from many writers, so a single reported accuracy can hide large failures on one person's
handwriting. We study whether a pretrained recognizer can be personalized to a new writer
from 10 to 100 labeled examples. Starting from CoMER, a Transformer-based HMER model with
attention coverage, we compare three adaptation methods: full fine-tuning, decoder-only
fine-tuning, and low-rank adaptation (LoRA) of the decoder feed-forward layers, which
trains under 1% of the parameters. For each writer we hold out a fixed, formula-disjoint
test set and measure exact expression match and token edit distance against the shared
base model, together with the drop in accuracy on a general benchmark (forgetting). The
base model reaches 62.6% exact match on CROHME 2019, and 39% of its errors are a single
symbol substitution, mostly between look-alike glyphs (C/c, X/x, 9/q, S/5), which is
the kind of ambiguity that depends on the writer. In a CPU pilot on ten hard examples, all
three methods fit the support set but lose 4–14 points on images from other writers. The new
errors mirror the support corrections, which suggests that adaptation shifts glyph
decisions toward one writer's style. The writer study on newly collected team handwriting
runs on the SJSU HPC, and we will report learning curves, method cost, forgetting, and a
categorized error analysis.

## 1. Introduction

- **Problem.** A student photographs a handwritten equation and wants LaTeX. General HMER
  models misread individual styles: one writer's `1` looks like `l`, another's `B` like `3`.
  Benchmark accuracy averages these failures away.
- **Question.** Does adapting a pretrained recognizer to writer *w*, using *k* labeled
  examples, improve recognition of *new* expressions by *w*? How does the gain change with
  *k* ∈ {10, 25, 50, 100}? Which update scheme gives the best accuracy per trainable
  parameter, and what does each cost in general recognition?
- **Why it is non-trivial.** With 10 examples, fine-tuning 6.4 M parameters can memorize
  the support set and still fail on new expressions, or damage general recognition.
- **Contributions.** (1) A reproducible personalization pipeline around the official CoMER
  weights; (2) a locked, formula-disjoint per-writer evaluation protocol; (3) a comparison
  of full, decoder, and LoRA adaptation in accuracy, cost, and forgetting; (4) an error
  analysis that separates symbol confusions from structural mistakes; (5) a demo that
  adapts to a user's handwriting.

## 2. Related Work (literature survey)

**Image-to-sequence HMER.** Zhang et al. [1] introduced *Watch, Attend and Parse* (WAP),
which treats HMER as image captioning: a convolutional encoder produces a feature map and
an attention-based recurrent decoder emits LaTeX tokens. WAP adds a *coverage* vector that
accumulates past attention so the decoder neither skips nor re-reads strokes. Its
successor DenseWAP [2] replaced the encoder with a DenseNet [3] and added multi-scale
attention for small symbols. CoMER keeps that DenseNet encoder.

**Transformer decoders.** BTTR [4] replaced the recurrent decoder with a Transformer [5]
and trained it bidirectionally (left-to-right and right-to-left), which helps because LaTeX
errors are often local but propagate through the sequence. Plain Transformer attention,
though, has no coverage mechanism.

**CoMER (our base model).** Zhao and Gao [6] reintroduced coverage into the Transformer
with an *attention refinement module* that uses past attention weights from the current
layer (self-coverage) and from previous layers (cross-coverage). It reported state-of-the-art
expression rates on CROHME 2014/2016/2019 and released code and weights, so it is a strong,
reproducible base for adaptation.

**Structure-aware recognition and larger data.** SAN [7] adds a syntax-aware decoder that
parses expressions as grammar trees, and introduced HME100K, a 100,000-image dataset of
camera-captured handwriting. It shows that real phone photos are a different domain from
the scanned CROHME images, which motivates our optional phone-domain stage.

**Benchmarks.** The CROHME competitions define the standard test sets. The 2023
edition [8] combines several sources and modalities and describes the official symbol-
and structure-level evaluation. We use CROHME 2014 and 2019 test sets as general
references and report our token-level metrics under their own names, not as official
CROHME scores.

**Writer adaptation.** Writer adaptation has been studied mostly for handwritten *text*.
MetaHTR [9] meta-learns an initialization so that a single gradient step on a few samples
from a new writer adapts the recognizer at inference time. We study the simpler,
widely available setting of fine-tuning a released model without meta-training, and
apply it to 2-D mathematical layout rather than text lines.

**Parameter-efficient fine-tuning.** Adapters [10] insert small trainable bottleneck
layers into a frozen network and match full fine-tuning on NLP tasks with about 3% extra
parameters per task. LoRA [11] instead adds a trainable low-rank update *BA* to frozen
weight matrices; *B* starts at zero, so training starts exactly at the base model. For
personalization this makes a per-user model a small file instead of a full copy. We test
whether LoRA, designed for language models, transfers to a small vision recognizer.

**Forgetting.** Fine-tuning on a narrow dataset can erase previously learned behavior
(catastrophic forgetting) [12]. We measure it directly as the drop in general-benchmark
accuracy while a personalized model is active.

*To confirm before the final report: none of [1]–[12] is on the class reading list.*

## 3. Technical Approach

### 3.1 Problem formulation

Let $x$ be an image of one handwritten expression and $y=(y_1,\dots,y_T)$ its LaTeX token
sequence over a vocabulary $\mathcal{V}$ ($|\mathcal{V}|=113$ including special tokens).
An autoregressive recognizer models

$$p_\theta(y\mid x)=\prod_{t=1}^{T}p_\theta(y_t\mid y_{<t},x).$$

For writer $w$ we have a support set $S_w^{(k)}$ of $k$ labeled images and a disjoint test
set $Q_w$ of 50 new formulas. Personalization starts from shared base parameters $\theta_0$
and finds $\theta_w^{(k)}$ using only $S_w^{(k)}$. Success means higher accuracy on $Q_w$
than $\theta_0$.

### 3.2 Base model

CoMER [6]: a DenseNet encoder (16 layers per block, growth rate 24) produces a feature map;
a 3-layer Transformer decoder ($d_\text{model}=256$, 8 heads, feed-forward width 1024)
with self- and cross-coverage attention emits tokens. We use the official released weights
at commit `ee15cdb`, converted without changing the architecture: **6,389,041 parameters**
(encoder 3,156,624; decoder 3,232,417). Decoding uses the official bidirectional beam search,
beam size 10.

### 3.3 Training objective

Teacher-forced negative log likelihood, averaged over non-padding tokens and computed in
both directions (left-to-right and reversed sequence $\overleftarrow{y}$), as in BTTR/CoMER:

$$\mathcal{L}(\theta)=-\frac{1}{N_\text{tok}}\sum_{i}\sum_{t}\Big[\log p_\theta(y_{i,t}\mid y_{i,<t},x_i)+\log p_\theta(\overleftarrow{y}_{i,t}\mid \overleftarrow{y}_{i,<t},x_i)\Big].$$

Optimizer AdamW (weight decay $10^{-4}$), gradient-norm clipping at 1.0, mini-batches of 4
sampled with replacement, a fixed update budget (200 steps by default). BatchNorm running
statistics stay frozen for every method, because 10 images are too few to re-estimate them.

### 3.4 Adaptation methods

| Method | What trains | Trainable parameters | Default LR |
|---|---|---|---|
| Full fine-tuning | all of $\theta$ | 6,389,041 (100%) | $10^{-5}$ |
| Decoder-only | $\theta_\text{decoder}$; encoder frozen, encoder dropout off | 3,232,417 (50.6%) | $5\times10^{-5}$ |
| LoRA (rank 8) | low-rank updates on 6 decoder FFN matrices | 61,440 (0.96%) | $10^{-3}$ |

**LoRA.** A frozen weight $W_0\in\mathbb{R}^{d_\text{out}\times d_\text{in}}$ becomes

$$h = W_0 z + \frac{\alpha}{r} B A z,\qquad A\in\mathbb{R}^{r\times d_\text{in}},\ B\in\mathbb{R}^{d_\text{out}\times r},$$

with $r=8$, $\alpha=16$, $B$ initialized to zero. Each adapted matrix adds
$r(d_\text{in}+d_\text{out})=8(256+1024)=10{,}240$ parameters instead of
$d_\text{in}d_\text{out}=262{,}144$. We target the two feed-forward matrices in each of
the 3 decoder layers (6 matrices, 61,440 parameters). CoMER's attention uses raw projection
tensors rather than separate `q_proj`/`v_proj` modules, so the usual attention-LoRA recipe
does not apply. We adapt the FFN layers instead and state this explicitly.

### 3.5 Preprocessing

Grayscale; each manifest row declares its polarity, and images are converted to CoMER's
white-ink-on-black representation. Aspect ratio is preserved, images are downscaled to fit
1024×256, and tiny images are padded. Optional training-only augmentation (mild blur,
illumination gradient, ±2% perspective jitter) is used only in the phone-domain stage.

### 3.6 Evaluation protocol

- **Data per evaluation writer:** at least 150 distinct formulas. 50 form a fixed test set;
  from the remaining 100, nested support prefixes of size 10 ⊂ 25 ⊂ 50 ⊂ 100 are drawn with a
  fixed seed. Every method and seed uses the same prefixes and the same test set.
- **Leakage control:** no normalized formula, image, or ID appears in both a writer's support
  and test files (checked by `wlm.leakage`).
- **Tuning:** learning rate and update budget are chosen on separate development writers;
  the evaluation writers' test sets are used only once, for final numbers.
- **Seeds:** 3 seeds per (writer, method, *k*), averaged within writer before aggregating.
- **General retention:** each adapted model is also evaluated on the CROHME 2019 test set
  (1,199 images, no writer IDs).

### 3.7 Metrics and statistics

Token-level metrics after conservative tokenization (`wlm/metrics.py`): **exact match**
(token edit distance 0); **within 1 / 2 tokens**; **mean edit distance (ED)**; **normalized
ED** $=\text{ED}/\max(|y|,|\hat y|,1)$; **token error rate** $=\sum\text{ED}/\sum|y|$.
These count LaTeX tokens, including braces. They are not the official CROHME symbol-graph
metrics.

- **Writer-macro accuracy:** the mean of per-writer accuracies, so each writer counts equally.
- **Paired gain:** adapted minus base exact match on the *same* test images, per writer.
- **Uncertainty:** 95% bootstrap intervals over writers (2,000 resamples), after averaging
  seeds within a writer. Seeds are not independent writers.
- **Paired significance on one test set:** an exact McNemar test on the images that the base
  and adapted models disagree on (base wrong → adapted right, *b*; and the reverse, *c*):
  $p = \min\!\big(1,\ 2\sum_{i=0}^{\min(b,c)}\binom{b+c}{i}2^{-(b+c)}\big)$.
- **Forgetting:** $100\times(\text{base general exact} - \text{adapted general exact})$
  percentage points; positive means a drop.
- **Error categories:** a token alignment labels each error as symbol substitution
  (e.g. `1 → l`), brace error, layout error (touching `^`, `_`, `\frac`, `\sqrt`), or a
  missing/extra symbol. We then review 20–30 changed cases by hand (`wlm.compare`).

### 3.8 Implementation

PyTorch 2.5.1. Python package `wlm/`: data preparation and audit (`prepare`), training
(`train`), decoding and metrics (`predict`, `metrics`), sweep planning and Slurm array jobs
(`sweep`, `slurm/`), analysis (`analyze`), and error review (`compare`). Gradio demo in
`app.py`. Runs are resumable, and every result file records the SHA-256 of its checkpoint
and data, so results from different bases or test sets cannot be mixed by accident.
Laptop: setup, evaluation, demo. SJSU HPC (Slurm, NVIDIA GPUs): the training sweep.

## 4. Experimental Results

### 4.1 Data

| Set | Images | Writer IDs | Use |
|---|---|---|---|
| CROHME 2019 test (bundled with CoMER) | 1,199 | none | general reference and forgetting |
| CROHME 2014 test (bundled with CoMER) | 986 | none | CPU pilot support and held-out images |
| Team handwriting (3 writers) | *being collected* | yes | personalization study |

### 4.2 Base model on CROHME 2019

Official CoMER weights, beam size 10, Mac CPU (Apple M1 Pro). Run twice from independent
clean setups with identical results. Decode time ≈1.3 s per image.

| Metric | Value |
|---|---|
| Images | 1,199 |
| Exact token match | **62.64%** |
| Within 1 token | 80.07% |
| Within 2 tokens | 86.16% |
| Mean edit distance | 1.267 tokens |
| Mean normalized ED | 0.066 |
| Token error rate | 7.57% |

| Subset (overlapping) | Images | Exact |
|---|---|---|
| Contains a fraction | 352 | 65.6% |
| Contains a superscript | 427 | 52.5% |
| Contains a radical | 210 | 62.9% |
| Long (≥ 50 tokens) | 23 | 8.7% |

The CoMER paper [6] reports expression rates from the official CROHME evaluator. Our
scorer is token-level, so we measure our own baseline and do not compare it to the paper's
number directly. *(To do: quote the paper's CROHME 2019 figure from its results table for
context.)* Superscripts and long expressions are the weakest subsets.

### 4.3 Error analysis of the base model

We aligned each wrong prediction to its ground truth and classified every token error
(Section 3.7).

| Error type (448 wrong predictions) | Token errors |
|---|---|
| Symbol substitution | 601 |
| Brace error | 381 |
| Missing symbol | 209 |
| Layout (`^`, `_`, `\frac`, `\sqrt`) | 178 |
| Extra symbol | 150 |

**39.1% of the wrong predictions (175/448) differ from the truth by a single symbol
substitution.** The most frequent substitutions are glyph-shape ambiguities: `C→c` (19),
`X→x` (16), `|→1` (11), `9→q` (10), `P→p` (10), `S→5` (9), `V→v` (8), `z→2` (7).
Case-only confusions account for 13.6% of all substitutions. These depend on an individual's
handwriting: whether a writer's capital *C* is larger than their lowercase *c*, or how they
close a *9*. This is the error class personalization should be able to fix, and we
will track these pairs in the writer study. Brace errors usually come with a layout
mistake (a misread superscript drops its `{ }`), which matches the low superscript accuracy
above.

*To do: manually tag the 30-case review sheet produced by `wlm.compare` and report the
agreement with the automatic types.*

### 4.4 Pilot: the three adaptation methods on CPU

Before collecting team data, we ran all three methods on the Mac CPU to check the full
train → decode → score pipeline and to see how each behaves with 10 examples. **This is not
a personalization experiment:** the 10 support images are CROHME 2014 test images from
many unknown writers, chosen because the base model gets all of them wrong (token ED 1–4).
We then decode 50 other 2014 images and 200 fixed CROHME 2019 images that none of the
models trained on. Settings: 200 updates, batch 4, seed 0, default learning rates.

| Model | Support (n=10) | Held-out 2014 (n=50) | General 2019 (n=200) |
|---|---|---|---|
| Base | 0.0% | 76.0% | 67.5% |
| Full fine-tuning | 100.0% | 62.0% | 63.5% |
| Decoder-only | 100.0% | 70.0% | 61.5% |
| LoRA (r=8) | 100.0% | 66.0% | 62.0% |

*Table: exact token match. Same 200-update budget for every method.*

| Method | Trainable parameters | Training time (CPU) | Stored size |
|---|---|---|---|
| Full | 6,389,041 | 84.8 s | 74.1 MiB |
| Decoder | 3,232,417 | 28.4 s | 49.8 MiB |
| LoRA | 61,440 | 25.5 s | 0.73 MiB |

*Table: method cost, 200 updates on an Apple M1 Pro CPU. Stored size includes optimizer state.*

**Observations.**

1. *All three methods work end to end and can fit a tiny support set* (0% → 100%), which
   rules out label, polarity, and gradient bugs before the HPC runs.
2. *All three hurt unseen images from other writers* (−6 to −14 points on held-out 2014,
   −4 to −6 on general 2019). With 50–200 test images only some drops are significant
   (exact McNemar test: e.g. LoRA on 2019 fixes 6 images and breaks 17, *p* = 0.035).
3. *The new errors mirror what the support set taught.* The support corrections included
   `M→m`, `5/s→S`, `.→,` and `3→B`. On other images, the most common new errors were
   `M→m`, `s→S`, `.→,` and `\beta→B`. Adaptation moves the boundary between ambiguous glyph
   pairs, the same pairs the base model confuses most (Section 4.3). For a single writer
   that is the intended effect; across writers it is interference. This argues for keeping
   one small, switchable adapter per writer, which is LoRA's practical advantage: 68× less
   storage than decoder tuning at similar training time.
4. *Fewer updates did not help.* LoRA at 50 and 100 updates already reaches 100% on the
   support set with the same held-out loss (66% held-out at every budget). The learning rate
   is the next lever, and our development tuning on the HPC will include lower rates.

What the pilot cannot tell us is whether adaptation helps on *new expressions by the same
writer*. That needs the team handwriting data and is the main experiment (4.5).

### 4.5 Writer study *(planned; runs on the SJSU HPC)*

- Base per writer on the frozen test set; 3 writers × 3 methods × 4 budgets × 3 seeds = 108 runs
- Figure: writer-macro exact match vs. *k*, one line per method, shared-base reference line,
  bootstrap band
- Table: method cost: trainable parameters, training time, peak GPU memory, checkpoint size
- Table: forgetting on CROHME 2019 per method and *k*
- Figure: paired image / base / adapted predictions for improved and regressed cases
- Optional: cross-writer control (adapt on writer B, test on writer A) to separate writer
  specificity from generic extra training

## 5. Discussion and Conclusion *(planned)*

What improved and for whom; whether LoRA's accuracy justifies its 100× smaller update;
limitations: 3 writers (exploratory, not a population claim), one capture setup, the
113-token vocabulary, token metrics rather than official CROHME scoring, and unknown
pretraining writer overlap for public data; next steps.

## 6. Work Division

| Member | Responsibility |
|---|---|
| Jeriel Zhi Hao Goh | Model setup, training code, all SJSU HPC runs |
| Aidana Muralieva | Team handwriting collection, labels, writer splits, dataset audit |
| Aaditya Desai | Evaluation protocol, metrics, error analysis, demo, report assembly |

## References

[1] J. Zhang, J. Du, S. Zhang, D. Liu, Y. Hu, J. Hu, S. Wei, and L. Dai, "Watch, attend and parse: An end-to-end neural network based approach to handwritten mathematical expression recognition," *Pattern Recognition*, vol. 71, pp. 196–206, 2017, doi: 10.1016/j.patcog.2017.06.017.

[2] J. Zhang, J. Du, and L. Dai, "Multi-scale attention with dense encoder for handwritten mathematical expression recognition," in *Proc. 24th Int. Conf. Pattern Recognition (ICPR)*, 2018, pp. 2245–2250, doi: 10.1109/ICPR.2018.8546031.

[3] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 4700–4708.

[4] W. Zhao, L. Gao, Z. Yan, S. Peng, L. Du, and Z. Zhang, "Handwritten mathematical expression recognition with bidirectionally trained Transformer," in *Proc. Int. Conf. Document Analysis and Recognition (ICDAR)*, 2021, pp. 570–584, doi: 10.1007/978-3-030-86331-9_37.

[5] A. Vaswani *et al.*, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 5998–6008.

[6] W. Zhao and L. Gao, "CoMER: Modeling coverage for Transformer-based handwritten mathematical expression recognition," in *Proc. European Conf. Computer Vision (ECCV)*, 2022, pp. 392–408, doi: 10.1007/978-3-031-19815-1_23.

[7] Y. Yuan *et al.*, "Syntax-aware network for handwritten mathematical expression recognition," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 4553–4562.

[8] Y. Xie *et al.*, "ICDAR 2023 CROHME: Competition on recognition of handwritten mathematical expressions," in *Proc. Int. Conf. Document Analysis and Recognition (ICDAR)*, 2023, pp. 553–565, doi: 10.1007/978-3-031-41679-8_33.

[9] A. K. Bhunia, S. Ghose, A. Kumar, P. N. Chowdhury, A. Sain, and Y.-Z. Song, "MetaHTR: Towards writer-adaptive handwritten text recognition," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 15830–15839, doi: 10.1109/CVPR46437.2021.01557.

[10] N. Houlsby *et al.*, "Parameter-efficient transfer learning for NLP," in *Proc. 36th Int. Conf. Machine Learning (ICML)*, PMLR vol. 97, 2019, pp. 2790–2799.

[11] E. J. Hu *et al.*, "LoRA: Low-rank adaptation of large language models," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2022.

[12] J. Kirkpatrick *et al.*, "Overcoming catastrophic forgetting in neural networks," *Proc. National Academy of Sciences*, vol. 114, no. 13, pp. 3521–3526, 2017, doi: 10.1073/pnas.1611835114.

**Software and data.** CoMER official code and weights, commit
`ee15cdbe02467fd29f386fd554b4bc2185110fcc`, <https://github.com/Green-Wood/CoMER>, accessed
2026-10-21. CROHME 2014/2019 test sets as bundled in that repository.
