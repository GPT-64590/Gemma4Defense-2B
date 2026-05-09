# Gemma4Defense-2B

[![Model Card](https://img.shields.io/badge/Model%20Card-Gemma4Defense--2B-yellow)](https://huggingface.co/athena129/Gemma4Defense-2B)
[![Companion](https://img.shields.io/badge/Companion-CyberSecQwen--4B-blue)](https://github.com/GPT-64590/CyberSecQwen-4B)
[![License: MIT (code)](https://img.shields.io/badge/license-MIT%20%28code%29-green.svg)](LICENSE)
[![Model License: Gemma](https://img.shields.io/badge/model-Gemma%20Terms-orange.svg)](https://ai.google.dev/gemma/terms)

A 2.3B-parameter cybersecurity language model fine-tuned from [Gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) for CWE classification (CTI-RCM) and cyber threat intelligence multiple-choice (CTI-MCQ). Under [Cisco's Foundation-Sec evaluation protocol (arXiv:2504.21039)](https://arxiv.org/abs/2504.21039), Gemma4Defense-2B retains 98.6% of Foundation-Sec-Instruct-8B's CTI-RCM accuracy at one-quarter the parameter count, and exceeds its CTI-MCQ by +10.5 points.

This repository contains everything needed to reproduce the model: training corpus assembly (with explicit decontamination), supervised fine-tuning, multi-trial evaluation, and the released benchmark numbers.

---

## Contents

- [Headline benchmark results](#headline-benchmark-results)
- [Quick start (inference)](#quick-start-inference)
- [Reproducibility](#reproducibility)
- [AMD MI300X — what we used and what we learned](#amd-mi300x--what-we-used-and-what-we-learned)
- [Repository structure](#repository-structure)
- [Methodology summary](#methodology-summary)
- [Limitations and intended use](#limitations-and-intended-use)
- [Citation](#citation)
- [License](#license)
- [Companion model](#companion-model)

---

## Headline benchmark results

5 trials per cell, temperature 0.3, no system prompt, dataset-`Prompt`-column-as-user-message. Mean ± standard deviation.

| Benchmark | Gemma4Defense-2B (2.3B) | Foundation-Sec-Instruct-8B | Δ vs target |
|---|---:|---:|---:|
| CTI-MCQ (2,500 items) | **0.6042 ± 0.0090** | 0.4996 | **+10.5 pp** |
| CTI-RCM (1,000 items) | **0.6754 ± 0.0035** | 0.6850 | -1.0 pp (within ~3σ) |

A companion model trained with the **same recipe** on Qwen3-4B-Instruct-2507 — [CyberSecQwen-4B](https://github.com/GPT-64590/CyberSecQwen-4B) — converges to within 0.9 points on CTI-RCM, demonstrating recipe portability across model families.

Full evaluation (more comparators, including independent reproduction of CyberPal-2.0-20B at our protocol): see [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md).

---

## Quick start (inference)

```bash
pip install transformers torch accelerate
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "athena129/Gemma4Defense-2B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

cve = ("A deserialization vulnerability in the destruct() function of Laravel "
       "v8.5.9 allows attackers to execute arbitrary commands.")

messages = [{
    "role": "user",
    "content": (
        "Analyze the following CVE description and map it to the appropriate CWE. "
        "Provide a brief justification for your choice. "
        "Ensure the last line of your response contains only the CWE ID.\n\n"
        f"CVE Description: {cve}"
    ),
}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## Reproducibility

Three commands, in order. Each is a thin wrapper around the underlying scripts in `src/`.

```bash
# 1. Assemble the training corpus (decontamination + composition).
#    Reads MITRE/NVD CVE→CWE mappings filtered to 2021-only with CTI-Bench
#    overlap items removed, plus synthetic CVE/CTI Q&A.
bash build_corpus.sh

# 2. Fine-tune Gemma-4-E2B-it via LoRA on the assembled corpus.
#    LoRA r=64, alpha=64, dropout=0.05, lr=5e-5, 10 epochs, bf16, sdpa.
bash train.sh

# 3. Evaluate the trained model under Cisco's Foundation-Sec protocol
#    on CTI-RCM and CTI-MCQ at 5 trials. Output: results/multi_trial_5x.json.
bash eval.sh
```

**System requirements** for full reproduction:
- 1× GPU with ≥ 24 GB VRAM (training) or ≥ 12 GB VRAM (inference only)
- Python 3.11+, PyTorch 2.6+, ROCm 7 (or an equivalent modern PyTorch GPU stack)
- ~50 GB disk for HF cache and intermediate artifacts

The released checkpoint was trained on AMD MI300X via the AMD Developer Cloud. Training is hardware-agnostic and runs on any modern data-center GPU with ≥ 24 GB VRAM. Note: FlashAttention-2 is **not** enabled for this model because Gemma-4 uses head_dim=512 on global-attention layers, which exceeds the AMD gfx942 LDS budget; we use PyTorch `sdpa` instead. The companion CyberSecQwen-4B model (Qwen3 head_dim=128) does enable FA2 and runs ~1.6× faster per training step.

---

## AMD MI300X — what we used and what we learned

The released Gemma4Defense-2B checkpoint was trained on a single AMD Instinct MI300X 192 GB instance via the AMD Developer Cloud (`atl1` region), using the official `vllm/vllm-openai-rocm:latest` Docker image with ROCm 7 + PyTorch 2.6 + Hugging Face transformers + PEFT + TRL.

### What works cleanly on Gemma-4 + AMD MI300X

- **PyTorch `sdpa` attention.** Stable, no special build steps needed — ships in the official vLLM ROCm image.
- **bfloat16 throughout.** Both training and inference. No precision-mode papercuts.
- **vLLM ROCm serving.** `--attention-backend TRITON_ATTN` is the recommended inference backend on MI300X for this architecture; we used it for evaluation runs.
- **AITER kernels enabled** (`VLLM_ROCM_USE_AITER=1`, `TORCH_BLAS_PREFER_HIPBLASLT=1`, `HIP_FORCE_DEV_KERNARG=1`) for matmul throughput.
- **Single-instance pipeline.** No multi-node, no special interconnect — the full SFT-merge-eval pipeline runs on one MI300X.

### What does NOT work: FlashAttention-2 on Gemma-4

We attempted to enable FlashAttention-2 via the Composable-Kernels backend on AMD `gfx942` and it fails on Gemma-4 specifically. The reason is hardware: Gemma-4 uses **dual head_dim per layer** (256 on sliding-attention layers, **512** on global-attention layers at indices `[4, 9, 14, 19, 24, 29, 34]`). FA2-CK on gfx942 is bounded at head_dim ≤ 256 by the LDS (shared-memory) budget on MI300X. The model loads with `attn_implementation="flash_attention_2"` but crashes at the first global layer's forward pass.

This was confirmed by Tri Dao (FA2 author) in [flash-attention#2427](https://github.com/Dao-AILab/flash-attention/issues/2427) — there is no current ROCm timeline for hdim>256 in Composable Kernels. **For Gemma-4 on MI300X, sdpa is the only working attention path.** Per-layer hybrid workarounds exist (FA2 on sliding layers, sdpa on global) but have a known attention-mask-shape bug at the time of this work.

The companion CyberSecQwen-4B model uses FA2 because Qwen3-4B's head_dim=128 fits within the LDS budget; that path runs ~1.6× faster per training step than the Gemma sdpa path on the same hardware.

### Multimodal-to-text-only weight extraction

Gemma-4 ships as a multimodal base with vision and audio towers. Standard `peft.merge_and_unload()` produces a checkpoint that includes `audio_tower` and `vision_tower` parameters — usable but bloated for text-only inference and confusing for downstream tooling. We extract the text-only language-model weights post-merge and re-publish a clean `Gemma4ForCausalLM` (model_type `gemma4_text`) that drops the multimodal towers but preserves the language model exactly. The released [HF checkpoint](https://huggingface.co/athena129/Gemma4Defense-2B) is the text-only variant.

### Hardware portability

The training recipe in `train.sh` is hardware-agnostic. To run on a non-AMD GPU stack, drop the AMD-specific environment variables (they're no-ops elsewhere) and use a regular Python venv instead of the vLLM ROCm Docker image. The 24 GB+ VRAM training minimum and 12 GB+ inference minimum apply equally on any vendor's hardware.

## Repository structure

```
Gemma4Defense-2B/
├── README.md                     # this file
├── LICENSE                       # MIT for code in this repo
├── CITATION.cff                  # GitHub-rendered citation
├── requirements.txt              # pinned Python dependencies
├── .gitignore
│
├── train.sh                      # single-command training reproducer
├── eval.sh                       # single-command 5-trial eval reproducer
├── build_corpus.sh               # single-command corpus assembly
│
├── src/
│   ├── train.py                  # LoRA SFT trainer (Gemma chat format, sdpa attention)
│   ├── build_corpus.py           # corpus decontamination + composition
│   └── cti_bench_eval.py         # Cisco-protocol benchmark harness
│
├── data/
│   ├── cti_bench/                # public eval data (TSV files)
│   │   ├── cti-rcm.tsv
│   │   └── cti-mcq.tsv
│   └── train/                    # training corpora (decontaminated)
│       ├── rcm_2021_train.jsonl  # CVE→CWE 2021 cohort, CTI-Bench overlap removed
│       └── cve_cti_synth.jsonl   # synthetic defensive-analyst Q&A
│
├── results/
│   ├── multi_trial_5x.json       # released benchmark numbers (5-trial mean ± std)
│   └── baseline_e2b_it.json      # Gemma-4-E2B-it raw baseline (pre-fine-tune)
│
└── docs/
    ├── RESEARCH_NOTES.md         # methodology, controlled experiments, lessons
    ├── RECIPE_PORTABILITY.md     # cross-substrate validation summary
    └── LIMITATIONS.md            # safety, ethics, abuse-prevention notes
```

---

## Methodology summary

This model uses **direct supervised fine-tuning (SFT)** of an instruction-tuned base via LoRA. Key design choices:

1. **Decontaminated training data.** An earlier internal iteration of this work showed roughly 72% test-set overlap when trained on undeduplicated CTI corpora. The released model trains exclusively on the 2021 CVE→CWE cohort with CTI-Bench overlap items removed, plus synthetic defensive-analyst Q&A grounded in CVE descriptions.
2. **Instruction-tuned base, not pre-trained base.** Direct SFT on the IT checkpoint preserves existing format priors (terse-answer multiple-choice convention) better than SFT on the pre-trained base. Comparable runs we conducted on `Gemma-4-E2B` (pretrained base) showed substantial CTI-MCQ format-binding decay (-39 pp in the worst case) at the same corpus scale. See [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md).
3. **Direct SFT, not knowledge distillation.** We evaluated knowledge-distillation variants from a 20B teacher model (CyberPal-2.0-20B) earlier in the project. At our corpus scale (~12K supervised records) direct SFT outperformed distillation on the headline benchmarks. The released model is direct SFT only.
4. **Multi-trial benchmarking.** All headline numbers are means of 5 independent trials with random sampling seeds at temperature 0.3; standard deviations are reported alongside.
5. **Cross-substrate validation.** The identical training corpus and hyperparameters were applied independently to Qwen3-4B-Instruct-2507 ([CyberSecQwen-4B](https://github.com/GPT-64590/CyberSecQwen-4B)). Both models converge within 0.9 points on CTI-RCM — built-in robustness check that the result is recipe-driven, not substrate-specific.

---

## Limitations and intended use

This is a defensive cybersecurity research artifact. It is not appropriate for:
- Generating exploit code, weaponized PoC, or attacker tradecraft
- Auto-executing security decisions without qualified human review
- Legal, medical, or regulated-advice contexts
- Tasks outside cybersecurity (general chat, code generation)

Full intended-use, out-of-scope-use, and limitations text is in the [Hugging Face model card](https://huggingface.co/athena129/Gemma4Defense-2B). Practical recommendations and recommended-use guardrails are in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

---

## Citation

```bibtex
@misc{gemma4defense2026,
  title  = {Gemma4Defense-2B: A Compact CTI Specialist Fine-Tuned from Gemma-4-E2B-it},
  author = {Mulia, Samuel},
  year   = {2026},
  publisher = {Hugging Face},
  url    = {https://huggingface.co/athena129/Gemma4Defense-2B}
}
```

The evaluation protocol is from [Foundation-Sec-8B (arXiv:2504.21039)](https://arxiv.org/abs/2504.21039); the benchmark is [CTI-Bench](https://github.com/xashru/cti-bench).

---

## License

- **Code in this repository:** MIT — see [`LICENSE`](LICENSE)
- **The fine-tuned model weights** (hosted on Hugging Face): Gemma Terms of Use — see https://ai.google.dev/gemma/terms

The model is a derivative of `google/gemma-4-E2B-it` and inherits Google's Gemma license. The training data (decontaminated 2021 CVE→CWE mappings) is derived from public MITRE/NVD records; the synthetic CVE/CTI Q&A in `data/train/cve_cti_synth.jsonl` is original and released under the same MIT license as the code.

---

## Companion model

[CyberSecQwen-4B](https://github.com/GPT-64590/CyberSecQwen-4B) — sister release on Qwen3-4B-Instruct-2507, Apache 2.0, validated end-to-end on AMD MI300X. Same training recipe; converges to RCM 0.6664 ± 0.0023 / MCQ 0.5868 ± 0.0029. Use that model when the Gemma terms are not a fit for your deployment.
