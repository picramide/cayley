# Runs of the controlled study and the files they must produce

Every table and figure of the results sections is produced by three scripts from the files
described here. If the files follow this format exactly, no manual step is needed between
the runs and the paper.

Two encoders, analysed separately:
- `roberta`: `roberta-base`
- `deberta`: `microsoft/deberta-v3-base`

Seven GLUE tasks: `cola`, `sst2`, `mrpc`, `qqp`, `stsb`, `rte`, `qnli` (no MNLI).

Estimated cost: ~200 GPU-hours (RoBERTa) + ~265 (DeBERTa; assumes 1.3× RoBERTa's time, to be
measured) ≈ **465 GPU-hours** on one V100. QQP alone accounts for about 80 of these.

## 0. Before any run (no training)

1. **DeBERTa mask implementation.** Pass the Cayley mask to `DebertaV2ForSequenceClassification`
   as a per-pair attention mask of shape [batch, L, L], combined (logical AND) with the padding
   mask. Check that the classifier reads only position 0 (`ContextPooler` uses
   `hidden_states[:, 0]`). Check also that `config.position_biased_input` is `false`, since the
   paper states that DeBERTa-v3 uses relative positions only.
2. **DeBERTa audit.** Run the same four checks as for RoBERTa, on DeBERTa-v3-base, in fp16 on the V100:
   - an all-ones mask gives exactly the dense logits, loss and gradients;
   - excluded pairs get zero probability in all 12 layers;
   - the masks are identical across tasks;
   - in the circulant test (60 tokens), changing token 45 leaves the logits unchanged and
     changing token 20 changes them.
   
   Save the result as `data/implementation_audit_deberta_fp16.json`. It goes into Appendix B.1.
3. **New RoBERTa masks.** Run the audit on `hypercube_gray` and `hypercube_hampath0–2`, for both models.
4. **DeBERTa lengths.** Run `diagnostics.py lengths` with the DeBERTa-v3 tokenizer and save the
   output as `data/length_stats_deberta.json`. Then run
   `python3 scripts/length_tables.py data/length_stats_deberta.json --suffix _deberta`.
5. **DeBERTa pretraining-objective head.** The released checkpoint contains a replaced-token-detection
   head (weights `mask_predictions.*`). Load it and check that, under dense attention and with
   15% unigram replacements, it detects the replaced tokens well above chance (report the
   AUROC). If it cannot be loaded or is at chance, tell me: Experiment 2 (2-i) is then reported
   for RoBERTa only, as the analysis plan allows.
6. **Timing.** Time one dense DeBERTa run each on CoLA and QNLI under P, and check that the dense
   model trains normally in fp16. If fp16 produces NaNs or a collapsed dense model, stop and
   tell me before running anything else.

## 1. Runs

All runs use the original protocol (P) unless stated, fp16, max length 128, and checkpoint
selection on a **held-out 10% of the training set**. The split is fixed, saved, stratified by
label, and the same for every run and **both models**. The GLUE development set is used only
for reporting. Seed *k* must give the same classifier initialisation and data order in every
condition.

Run the following **for each model** (`roberta`, then `deberta`):

| Order | Experiment | Task | Mask | Conditions | Seeds |
|---|---|---|---|---|---|
| 1 | exp1 | cola | hypercube, dense | P | 1–20 |
| 1 | exp1 | qnli | window_dilations, local, dense | P | 1–5 |
| 1 | exp1 | rte | local, dense | P | 1–20 |
| 2 | exp2 (2-i) | — | dense + 5 placements | objective loss, no training | — |
| 2 | diagnostics | cola, sst2, qnli | 7 sparse masks | forward passes | — |
| 3 | exp1 | cola | hypercube | S, S+, G | 1–20 |
| 3 | exp1 | cola | dense | S, S+ | 1–20 |
| 3 | exp1 | qnli | window_dilations | S, S+, G | 1–5 |
| 3 | exp1 | qnli | dense | S, S+ | 1–5 |
| 3 | exp1 | rte | local | S, S+, G | 1–20 |
| 3 | exp1 | rte | dense | S, S+ | 1–20 |
| 4 | exp1 | cola / qnli / rte | hypercube / window_dilations / local | P in **fp32**, seed of the first degenerate fp16 run (if any) | 1 each |
| 5 | exp2 | cola | hypercube_gray, hypercube_hampath0–2 | P | 1–20 |
| 5 | exp2 | sst2 | dense, hypercube, hypercube_gray, hypercube_hampath0–2 | P | 1–5 |
| 5 | exp2 | mrpc | dense, hypercube, hypercube_gray, hypercube_hampath0–2 | P | 1–10 |
| 5 | exp2 | stsb | dense, hypercube, hypercube_gray, hypercube_hampath0–2 | P | 1–10 |
| 5 | exp2 | rte | hypercube, hypercube_gray, hypercube_hampath0–2 | P | 1–10 |
| 5 | exp2 | qnli | hypercube, hypercube_gray, hypercube_hampath0–2 | P | 1–5 |
| 6 | exp2 | qqp | dense, hypercube, hypercube_gray (**only these three**) | P | 1–3 |

The binary hypercube on CoLA and the dense runs on CoLA, QNLI and RTE are shared between the two
experiments: run them once, in Experiment 1. Run QQP last; it is the most expensive.

Select checkpoints on each task's primary metric: MCC (CoLA), F1 (MRPC, QQP), Pearson (STS-B),
accuracy (SST-2, RTE, QNLI).

Step 1 for DeBERTa tells us whether the same configurations fail at all in DeBERTa. Send me
those results before step 3 for DeBERTa. If DeBERTa shows no degeneracy, the plan still
applies: the protocols are then compared on mean score.

Conditions:
- **P**: original protocol (no warmup, 5 epochs).
- **S**: P plus linear warmup over the first 10% of the steps; same number of steps.
- **S+**: S with 15 epochs.
- **G**: S, plus a logit bias of −β(t) on the removed pairs. β rises linearly over the first
  half of training to β_max = 20, after which the hard mask is used. Padding is masked throughout.

## 2. Files

```
runs/
  results.jsonl
  <run_dir>/dev_predictions.jsonl
  <run_dir>/trainer_state.json
  objective_loss.jsonl
  diagnostics.json
```

**results.jsonl** — one line per fine-tuning run:
```json
{"model": "deberta", "experiment": "exp1", "task": "cola", "mask": "hypercube", "condition": "P",
 "seed": 3, "precision": "fp16", "run_dir": "deberta/exp1/cola/hypercube/P/seed3", "train_runtime_s": 512.4}
```
Allowed values:
- `model`: `roberta`, `deberta`
- `experiment`: `exp1`, `exp2`
- `task`: `cola`, `sst2`, `mrpc`, `qqp`, `stsb`, `rte`, `qnli`
- `mask`: `dense`, `local`, `window_dilations`, `hypercube`, `hypercube_gray`,
  `hypercube_hampath0`, `hypercube_hampath1`, `hypercube_hampath2`
- `condition`: `P`, `S`, `S+`, `G`
- `precision`: `fp16`, `fp32`

**dev_predictions.jsonl** — one line per GLUE development example, from the selected checkpoint:
```json
{"idx": 0, "label": 1, "pred": 1, "length": 12}
```
`length` is the number of non-padded tokens under **that model's tokenizer**, special tokens included.
For STS-B, `label` and `pred` are the gold and predicted similarity scores (floats); for all other
tasks they are class indices (MRPC: 1 = paraphrase; QQP: 1 = duplicate).

**trainer_state.json** — as written by the Trainer. Log the training loss at least every 0.01
epochs (CoLA: every 9 steps; SST-2: every 75; MRPC: every 4; STS-B: every 6; RTE: every 2;
QNLI: every 110; QQP: every 400).

**objective_loss.jsonl** — Experiment 2 (2-i). Use 128-token windows of the WikiText-103
validation set, all windows, each model with its own tokenizer. Write one line per
(model, window, mask):
```json
{"model": "roberta", "window": 17, "mask": "hypercube_gray", "loss_sum": 41.73, "n_tokens": 19}
```
- RoBERTa: masked-LM loss. Mask 15% of the tokens with the standard 80/10/10 procedure, with a
  fixed seed. `loss_sum` is the cross-entropy summed over the masked tokens; `n_tokens` is the
  number of masked tokens.
- DeBERTa: replaced-token-detection loss. Replace 15% of the tokens by tokens sampled from the
  unigram frequency of the DeBERTa vocabulary on the WikiText-103 training set, with a fixed
  seed. `loss_sum` is the binary cross-entropy summed over all non-special tokens; `n_tokens`
  is their number.

In both cases the masked or replaced positions must be identical for every mask of that model.

**diagnostics.json** — for each model, sparse mask and task, on 512 development inputs, before fine-tuning:
```json
{"deberta": {"hypercube": {"cola": {"retained": [12 floats], "single": [12 floats],
                                     "accumulated": 0.41, "readout": 0.37}, ...}, ...}, "roberta": {...}}
```
- `retained`: attention mass of the dense model that falls on the mask's edges, per layer.
- `single`: ‖o_masked − o_dense‖ / ‖o_dense‖ after the attention output projection, one layer
  masked at a time, on the dense model's inputs to that layer.
- `accumulated`: the same ratio for the final hidden states with all layers masked.
- `readout`: the same ratio for the position-0 representation.

## 3. Producing the tables and figures

```bash
python3 scripts/analyse_exp1.py  runs/results.jsonl
python3 scripts/analyse_exp2.py  runs/results.jsonl runs/objective_loss.jsonl runs/diagnostics.json
python3 scripts/analyse_extra.py runs/results.jsonl
```

These write the following, and the paper includes them automatically:
- `tables/exp1_*.tex`, `tables/exp2_*.tex`, `tables/diag.tex`, `tables/length_strata.tex`,
  `tables/compute_controlled.tex`
- `figures/exp1.pdf`, `figures/exp2.pdf`
- `data/exp1_summary.json`, `data/exp2_summary.json`

The result sentences in the paper are then written from the two summary files.

To test the pipeline without real runs (the data are fabricated, and the scripts refuse to
write them into the paper folder):
```bash
python3 scripts/make_test_runs.py /tmp/test_runs/in
python3 scripts/analyse_exp1.py /tmp/test_runs/in/results.jsonl --out /tmp/test_runs/out --B 1000
python3 scripts/analyse_exp2.py /tmp/test_runs/in/results.jsonl /tmp/test_runs/in/objective_loss.jsonl \
        /tmp/test_runs/in/diagnostics.json --out /tmp/test_runs/out --B 1000
```
