# AI Brand Mention Study

Replication data and code for the empirical study in **Chapter 4** of the bachelor thesis *Generative Engine Optimization: Measuring and Optimizing Brand Visibility in AI-Mediated Consumer Search* (BEMACS, Bocconi University).

The study asks **what predicts whether a brand is named in an AI answer**. Sixty consumer-style queries are put to two generative engines, brand mentions are extracted, and the mentions are modeled with a query-fixed-effects (conditional) logit against four public brand signals.

## What's in the study

- **60 queries**, 20 per purchase intent (Price, Security, Fit/Use). No query contains a brand name.
- **2 engines:** ChatGPT (OpenAI API, `gpt-5-mini`) and Gemini (Google Gemini API, `gemini-2.5-flash`).
- **7 brands:** 1Password, Bitwarden, LastPass, Dashlane, Keeper, NordPass, RoboForm.
- **840 observations** (60 queries × 2 engines × 7 brands), one binary `mention` per row.

## Repository structure

```
data/
  queries.csv                 60 queries with purchase intent
  responses_chatgpt.csv        raw ChatGPT answers (verbatim)
  responses_gemini.csv         raw Gemini answers (verbatim)
  mentions_chatgpt.csv         extracted binary mentions (420 rows)
  mentions_gemini.csv          extracted binary mentions (420 rows)
  data.csv                     modeling dataset (840 rows): mention + raw signals
code/
  run_chatgpt_queries.py       collect ChatGPT answers
  run_gemini_queries.py        collect Gemini answers
  extract_mentions.py          regex brand extraction
  dataset_constructor.py       merge mentions + signals into data.csv
  analysis.do                  original Stata estimation
  reproduce.py                 self-contained Python reproduction (no Stata needed)
output/
  regression_output.txt        reproduced full regression table, pseudo-R2, LR test
README.md
```

## Data collection protocol

Each query was sent **once to each engine** (120 calls total), as a **single user message** with **no system prompt** and the provider's **default sampling parameters** (no temperature, top-p, or max-token limit set on the generation call). **No web-search or browsing tool was enabled**, so answers reflect each model's parametric behavior at call time. Raw answers are stored verbatim with line breaks collapsed to spaces.

## Brand extraction

`extract_mentions.py` applies one case-insensitive regular expression per brand, allowing an optional internal space (e.g. `\b1 ?password\b`), producing one row per (query × brand) with a binary `mention`.

## Model

A conditional (fixed-effects) logit of the binary mention indicator, with **one group per query** and a platform indicator for Gemini, with **standard errors clustered by query**. The four signals are standardized (z-scores over all 840 rows):

- **Social buzz** = std( ln(1 + Reddit + YouTube + LinkedIn topic hits) )
- **Articles / PR** = std( ln(1 + listicle + own-domain topic hits) )
- **Review index** = std( G2 average rating × ln(1 + G2 review volume) )
- **SEO** = std( Lighthouse homepage score )

## Reproducing the results

No Stata required — `reproduce.py` rebuilds the features and refits the model in Python.

```bash
pip install pandas numpy scipy statsmodels
cd code
python reproduce.py
```

Stata users can run `analysis.do` instead (it expects the dataset as a local CSV).

### Key results

| Predictor | Odds ratio | p |
|---|---|---|
| Social buzz | 10.99 | < 0.001 |
| Articles / PR | 1.86 | < 0.001 |
| SEO score | 1.43 | 0.041 |
| Review index | 0.73 | 0.125 (n.s.) |
| Platform: Gemini (vs ChatGPT) | 0.41 | < 0.001 |

McFadden pseudo-R² ≈ 0.50. A likelihood-ratio test for equal marketing slopes across engines does not reject equality (LR = 3.08, df = 4, p = 0.55).

## Requirements

Python 3.10+ with `pandas`, `numpy`, `scipy`, `statsmodels`. The collection scripts additionally require `openai` and `google-genai` and read API keys from the `OPENAI_API_KEY` / `GOOGLE_API_KEY` environment variables.

## Notes

- **No API keys are stored in this repository.** The collection scripts read keys from environment variables; set them before running.
- Per-call collection timestamps were not recorded, and full Lighthouse HTML reports are not included (only the numeric homepage score, in `data.csv`).
- The data concern a public consumer market (password managers) and contain no proprietary or client information.

## Citation

> Ferraiolo, A. (2026). *Generative Engine Optimization: Measuring and Optimizing Brand Visibility in AI-Mediated Consumer Search* (Bachelor thesis, Bocconi University). Replication materials. GitHub.

## License

Released for academic use.
