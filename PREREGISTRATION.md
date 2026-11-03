# Preregistration

**Project:** Chain-of-thought vs. direct-answer prompting on GSM8K
**Author:** John Fillingim (AIPI 510, Duke)
**Date:** 2026-11-02. Committed before main data collection. The pilot is used only to freeze the prompts and the answer extractor and to recompute power.

## Research question

Does chain-of-thought (CoT) prompting improve LLM accuracy on GSM8K math word problems compared with direct-answer prompting, and is the gain worth the extra output tokens?

## Design

- **2 × 2 within-subjects.** Prompt style (`direct`, `cot`) × model (small = `qwen2.5:1.5b`, large = `qwen2.5:7b`). Every problem runs under all four conditions.
- **Sample.** We take one seeded permutation (seed 510) of the GSM8K test split (`openai/gsm8k`, config `main`, 1,319 problems). The first 200 problems are the main sample. The next 30 are a disjoint pilot sample, so the prompts are never tuned on test items.
- **Jobs.** 200 × 2 × 2 = 800 generations.
- **Controls.**
  - Temperature 0, seed 510, `num_predict` 512, `num_ctx` 4096.
  - The same system message for every job.
  - Model digests pinned in `config.yaml` and checked before each run.
- **Run order.** Models run in blocks (small, then large), because the 4 GB GPU cannot hold both models. Within each block, the problem × prompt order is shuffled with seed 510. With greedy decoding and a fixed seed, outputs do not depend on order, so order affects only latency.
- **Prompts.** Defined in `config.yaml` and frozen after the pilot. Direct asks for only the final number. CoT asks for step-by-step reasoning, then the final number after `####`.

## Outcomes

- `correct` (binary). The extracted answer equals the gold answer numerically (tolerance 1e-6).
- **Extraction rule.**
  - Use the number after the last `####` if one is present. Otherwise use the last number in the response.
  - Commas, `$`, and units are stripped.
  - If no number can be extracted, the response is scored incorrect and `parse_failed = True`.
- `output_tokens`. Ollama's `eval_count`.

## Hypotheses and tests

All tests are paired, by problem.

- **H1.** Within each model, CoT accuracy differs from Direct accuracy.
  - Test: exact McNemar test. Bonferroni α = 0.025 per model (two tests).
  - Report: accuracy difference with a paired Newcombe 95% CI, and the discordant-pair odds ratio (b/c) with an exact 95% CI.
- **Omnibus.** Cochran's Q across the four cells, then pairwise exact McNemar tests with Holm correction.
- **H2.** The CoT gain differs by model, i.e. a prompt × model interaction.
  - Primary test: conditional logistic regression grouped by problem, `correct ~ prompt * model`. This is the fixed-effects counterpart of `(1 | problem)`.
  - Report: odds ratios with 95% CIs, and a likelihood-ratio test of the interaction (1 df).
  - Robustness: GEE logistic model (exchangeable correlation, clustered by problem) with a Wald test of the interaction.
  - Why conditional logit: no Python library offers a maximum-likelihood GLMM with an LRT.
- **H3.** CoT produces more output tokens than Direct.
  - Test: Wilcoxon signed-rank, per model.
  - Report: median paired difference, Hodges–Lehmann 95% CI, and matched-pairs rank-biserial r.

## Headline metric

**Cost per correct answer** = total output tokens / number correct, computed per condition. 95% CI from a bootstrap of problems (10,000 resamples, seed 510).

## Power

Connor (1987) formula for McNemar, with p10 = 0.15, p01 = 0.05, two-sided α = 0.025, and power = 0.80. This gives **n = 188**, so n = 200. We recompute n from the pilot's observed discordance and report it. The sample size will not change without a documented deviation.

## Robustness checks

1. H1 rerun with parse failures excluded.
2. Bootstrap 95% CI (10,000 problem resamples) for the accuracy difference.
3. Count of truncated responses (those that hit 512 tokens), reported per condition.

## Exploratory (labelled as such)

CoT gain by solution step count (the number of lines in the GSM8K reference solution).

## Exclusions

None. Every generated response is analysed. Parse failures count as incorrect, except in robustness check 1.

## Deviations

Any change after this file is committed will be listed here with the date and the reason.
