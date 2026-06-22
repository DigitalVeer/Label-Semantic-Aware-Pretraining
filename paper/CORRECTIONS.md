# Corrections applied to the paper

The repository originally contained only the compiled `paper.pdf` (built with
pdfTeX on 2023-05-20); the LaTeX source was never committed. `paper/paper.tex`
is an **editable reconstruction** of that PDF with the corrections below
applied. Each change is also tagged inline in `paper.tex` with a
`% CORRECTED:` comment.

> Figures 1–8 are raster images that are not present in the repo. They are
> reproduced in `paper.tex` as captioned placeholders. Drop the originals into
> `paper/figures/` and replace the placeholders to restore them.

## Substantive validity fixes

1. **Misattribution of LSAP (most important).**
   Section 6 (“Approach”) originally read *“we propose a novel approach called
   Label Semantic Aware Pre-training (LSAP)…”*. LSAP is **not** the authors’
   contribution — it is from Mueller & Roth (2022), as the Introduction itself
   states. Reframed throughout as a **replication and analysis study**; the
   abstract now states this explicitly.

2. **Central self-contradiction about the best model.**
   The original simultaneously called Model 3 (LSAP) *“our best model”* and, in
   §6.2, stated that Model 1 *“consistently demonstrates superior performance
   across all evaluation measures for the ATIS dataset.”* These cannot both be
   true. Rewritten to present the genuine, reconcilable finding:
   - Model 3 (LSAP) tends to lead on **few-shot accuracy** (Tables of ATIS/SNIPS
     eval accuracies), reproducing the original trend.
   - Model 1 (plain T5-small) leads on **generation-quality metrics** (BLEU,
     cosine, Jaccard) for ATIS.
   - Added an explanation of *why* (surface-overlap metrics reward fluent,
     unadapted label generation), and softened the conclusion so LSAP’s benefit
     is described as a narrow low-shot accuracy effect at this scale.

3. **Cancelled / failed runs presented as results.**
   Many `0.0` cells are cancelled (OOM/time-limit) runs or fully failed models,
   not measured zero performance. Added a prominent **“Note on 0.00 entries”**
   paragraph, marked affected full-resource cells with `†`, and labeled Models
   4/5 rows as failed runs in every auxiliary-metric table.

4. **Reported tables don’t match committed data.**
   The accuracy values in the PDF do not exactly match the per-model CSVs under
   `analysis/` (e.g., `analysis/ATIS/acc.csv` row 1 one-shot = 65.1 vs. paper
   Table 5 Model 1 one-shot = 51.03 — different runs). Added a
   **“Reproducibility caveat”** stating the committed CSVs are canonical and the
   tables should be regenerated from them.

5. **Models 4/5 framing.**
   The from-scratch (random-init) ablation produced 0% everywhere. The Baselines
   section now flags up front that this ablation failed at our scale, rather than
   implying usable results.

## Typo / consistency fixes

6. **Duplicate label in Error Analysis.** *“predicted as either
   ‘SearchScreeningEvent’ or ‘SearchScreeningEvent’”* → *“… or
   ‘SearchCreativeWork’,”* matching the rest of the error discussion.

7. **“Awware Pretraining”** in the Conclusion → **“Aware Pre-training.”**

8. **Corpus-size wording** made internally consistent with Table 1
   (10,003 + ~110,000 ≈ ~120,000 examples).

9. **Learning-rate inconsistency.** The single rate ($5\times10^{-3}$,
   secondary pre-training) vs. the tuning range ($4\times10^{-3}$–$2\times10^{-2}$,
   fine-tuning) are now disambiguated, with a note that both are high for T5 and
   contributed to run-to-run variance.

## Not changed (left to the authors)

- **Re-running experiments** to resolve the table-vs-CSV mismatch or to obtain
  real full-resource / Models 4–5 numbers. The text now flags these honestly;
  it does not invent numbers.
- **Exact figure images** — placeholders only, since the sources aren’t in the
  repo.
