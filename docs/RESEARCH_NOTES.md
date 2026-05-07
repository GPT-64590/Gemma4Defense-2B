# Research notes — Gemma4Defense-2B

This document captures the methodology, controlled-experiment design, and key learnings from the development of Gemma4Defense-2B. The final released model is the result of an experiment series that explored several design dimensions before settling on the recipe shipped here.

## Goal

Build a compact (≤ 4B parameter) open-weight cybersecurity language model that:
- Maps CVE descriptions to CWE categories well (CTI-RCM)
- Answers multiple-choice cyber threat intelligence questions (CTI-MCQ)
- Performs comparably to Cisco's Foundation-Sec-Instruct-8B under the same evaluation protocol

## Evaluation protocol

We use the protocol described in [Foundation-Sec-8B (arXiv:2504.21039) §B.3-B.4](https://arxiv.org/abs/2504.21039):
- **IFT models** (instruction-tuned, including ours): zero-shot, the dataset's `Prompt` column as the user message, no system prompt
- **Pretrained base models**: 5-shot, exemplars sampled from the same TSV (Cisco's choice since CTI-Bench has no held-out dev split), prefix sentence, "Answer: X" format
- Temperature 0.3 across both modes
- Concurrency 32 against a vLLM-hosted endpoint
- 5 independent trials with random sampling seeds; mean + standard deviation reported

The eval harness is in `src/cti_bench_eval.py`; the inputs are CTI-Bench's TSV files committed under `data/cti_bench/`.

## Decontamination methodology

An earlier internal version (informally referred to as "v3") trained on undeduplicated public CTI corpora produced inflated CTI-RCM scores. Checking sample-level overlap revealed approximately **72% of training items appeared verbatim or near-verbatim in CTI-Bench's evaluation TSV**.

The released v3.4 (Gemma4Defense-2B) addresses this by:
1. Restricting the CWE classification training data to **MITRE/NVD records dated 2021** (the cti-rcm-2021 cohort), filtered against CTI-Bench's full RCM evaluation split with overlap items explicitly removed
2. Using the resulting `data/train/rcm_2021_train.jsonl` (6,776 records) as the primary specialization signal
3. Augmenting with synthetic defensive-analyst Q&A grounded in CVE descriptions but distinct from CTI-Bench items: `data/train/cve_cti_synth.jsonl` (~5,776 records)

The released training corpus (~12,500 supervised records) ships in `data/train/` so judges and reviewers can independently verify decontamination by comparing against `data/cti_bench/cti-rcm.tsv`.

## Why instruction-tuned base, not pretrained base

In an early experiment we trained `google/gemma-4-E2B` (the pretrained base, no instruction tuning) on the same corpus. Result:
- CTI-RCM lifted +15 pp (good)
- **CTI-MCQ collapsed from 0.570 (raw base 5-shot) to 0.182** (training-format prior never built)

The instruction-tuned variant `google/gemma-4-E2B-it`:
- CTI-RCM lifts comparably (~+9 pp)
- CTI-MCQ is *preserved* at the IT base's level (small lift)

Mechanism: at our corpus scale (~12K records), the pretrained base never sees enough multiple-choice format to retain it under SFT. The IT base's pretraining already includes terse-answer MCQ patterns; SFT on RCM-heavy data on top of IT preserves those priors rather than displacing them.

The released model uses `google/gemma-4-E2B-it`. The pretrained-base experiment is preserved as a negative result — see also `docs/RECIPE_PORTABILITY.md` for the cross-substrate replication of this finding.

## Direct SFT, not knowledge distillation

We evaluated CoT-trace knowledge distillation from a 20B teacher model ([CyberPal-2.0-20B](https://huggingface.co/cyber-pal-security/CyberOss-2.0-20B)) earlier in development. The distillation pipeline produced ~4,000 GT-correct CoT traces from the teacher, mixed with multi-task rehearsal data, and trained Gemma-4-E4B-base on the result.

Outcomes at our corpus scale:
- Direct SFT on the 12K decontaminated corpus yielded CTI-RCM 0.6754 / CTI-MCQ 0.6042
- CoT distillation on a similar-size mix yielded CTI-RCM 0.656 / CTI-MCQ 0.511 (different recipe variants tried; this is the best of them)
- Direct SFT outperformed distillation on both subsets

Hypothesis: at the small-corpus regime tested here, the teacher's CoT traces add reasoning-style content that doesn't compensate for the format-specialization signal lost when the training data becomes more heterogeneous. CyberPal's published recipe (arXiv:2510.14113) succeeds at much larger scale (SecKnowledge 2.0 training corpus); replicating that scale was outside this work's compute budget.

The released model is direct SFT only.

## Cumulative epoch progression (the actual training history)

The released v3.4 weights were produced through cumulative LoRA adapter resumption rather than a single 10-epoch run. The progression:

| Stage | Corpus | Epochs added | Cumulative epochs | CTI-RCM | CTI-MCQ |
|---|---|---:|---:|---:|---:|
| v3.1 | rcm_2021 only | 5 | 5 | 0.637 | 0.616 |
| v3.2 | + cve_cti_synth | +2 | 7 | 0.652 | 0.616 |
| v3.3 | combined | +3 | 10 (effective) | 0.665 | 0.612 |
| v3.4 | combined | +5 | 15 (released) | 0.6754 | 0.6042 |

(Single-trial intermediate numbers above; multi-trial available only for v3.4 in `results/multi_trial_5x.json`.)

The single-run recipe in `train.sh` (10 epochs on the combined corpus) is a *reproducible equivalent* but does not exactly recreate the cumulative training history. Cross-validation against the Qwen3 substrate (where we did run a single-shot 10-epoch recipe) shows the two paths converge to within multi-trial measurement noise on CTI-RCM. See [`docs/RECIPE_PORTABILITY.md`](RECIPE_PORTABILITY.md).

## Multi-trial validation

The headline numbers are 5-trial means, not single-trial measurements. Empirically, our recipe + corpus + sampling regime produces tight standard deviations (~0.005 RCM, ~0.009 MCQ at 5 trials), meaning headline claims are stable to within ~0.5 pp:

```
v3.4 (Gemma4Defense-2B):
  CTI-RCM: 0.6754 ± 0.0035  (5 trials)
  CTI-MCQ: 0.6042 ± 0.0090  (5 trials)
```

Single-trial measurements were within 1 pp of the 5-trial mean for all 4 cells we re-measured, but multi-trial averaging is the correct rigor level for any headline claim that compares against a published number with std-dev.

## Comparison to other models we evaluated

All numbers below are from our own measurement under the protocol described above.

| Model | Size | CTI-RCM | CTI-MCQ | Notes |
|---|---:|---:|---:|---|
| Foundation-Sec-8B (base) | 8B | 0.745 | 0.655 | 5-shot pretrained reference |
| **Foundation-Sec-Instruct-8B** | 8B | **0.685** | **0.500** | 0-shot, our TARGET |
| CyberPal-2.0-20B | 20B | 0.728* | 0.738* | independently verified at our protocol; their paper claims 0.874 / 0.757 with a different prompt template |
| **Gemma4Defense-2B** (this release) | 2.3B | **0.6754 ± 0.0035** | **0.6042 ± 0.0090** | 5-trial mean ± std |
| CyberSecQwen-4B (companion) | 4B | 0.6664 ± 0.0023 | 0.5868 ± 0.0029 | same recipe, different substrate |
| Gemma-4-E4B-it (raw) | 5.1B effective | 0.618 | 0.666 | 0-shot |
| Gemma-4-E2B-it (raw) | 2.3B | 0.580 | 0.578 | 0-shot, our base |
| Gemma-4-E4B-base (raw) | 5.1B effective | 0.588 | 0.666 | 5-shot |
| Gemma-4-E2B-base (raw) | 2.3B | 0.490 | 0.570 | 5-shot |

\* Single-trial values from our independent reproduction.

## What we tried that didn't work

For honesty, several recipe variants were trained and rejected:
- **Pretrained base + RCM-only single-task SFT** (v4.1 region): CTI-MCQ collapsed to 0.18
- **Pretrained base + multi-task CoT distillation** (v5.0 region): partial recovery to 0.51, still below released model
- **Multi-task corpus heavy on CoT, lighter on instruct format** (v6.0 region on Qwen): same MCQ-collapse pattern as v4.1
- **Doubling the rehearsal proportion in the multi-task mix** (v6.1 region on Qwen): MCQ partially recovered but RCM ceiling held at ~0.61

The single, robust finding across all these variants: **at our corpus scale, the IT base + direct SFT recipe shipped here is the strongest configuration we found.** The companion Qwen3 substrate experiment (CyberSecQwen-4B) confirmed this pattern is recipe-driven, not Gemma-specific.
