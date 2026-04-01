# scRT-agent v2

`scRT-agent v2` is a research-oriented agent for integrated `scRNA-seq + scTCR` analysis, designed to behave more like a research scientist than a workflow runner.

It is designed for datasets where:
- the transcriptome lives in a processed `AnnData` object (`.h5ad`)
- the TCR annotations live in a table such as `.csv`, `.tsv`, or `.txt`

The primary non-data input is a freeform `research brief`, not a paper. The brief can be plain notes, bullets, or rough prose. It does not need to follow a rigid template.

Compared with v1, v2 adds:
- sample-aware clonotype validation and automatic clonotype-scope repair inside the notebook setup
- a dataset validator that surfaces guardrails before planning
- an evidence ledger that tracks what each step actually supported, weakened, or left unresolved
- research-oriented planning fields such as `priority_question`, `evidence_goal`, `decision_rationale`, and `validation_checks`
- a fixed notebook helper library for paired-subset analyses and tissue-stratified differential expression
- a local literature reader and summarizer for PDF/markdown/txt inputs
- a literature-guided hypothesis candidate layer that ranks dataset-feasible mechanisms before notebook planning
- a publication-style figure mode for exporting multi-panel summary figures
- an interactive `prepare -> review -> run` CLI so users can revise hypotheses before notebook execution

The agent now:
1. summarizes the RNA object and TCR table
2. optionally reads local literature files and summarizes them for planning
3. turns the literature into ranked candidate hypotheses that are filtered by dataset feasibility
4. validates whether the inputs support the intended claims
5. selects or refines the best next hypothesis and priority question
6. writes code for the next notebook step
7. executes that code in a live Jupyter kernel
8. interprets the outputs
9. updates the evidence ledger
10. replans the next step
11. repeats until the analysis completes

## Main Inputs

- `--rna-h5ad-path`: processed scRNA `.h5ad`
- `--tcr-path`: scTCR annotation table
- `--research-brief-path`: freeform text file describing the research question, background, priorities, and any caveats
- `--literature-path`: optional local paper, review, note, or directory path; can be repeated

## Research Brief Guidance

Write the brief however you normally think about a project. A short note is enough if it is informative.

Useful things to mention when available:
- what biological question you care about
- what the samples or tissues mean
- what comparison matters most
- any known biology, markers, pathways, or expectations
- what would count as a useful result
- any claims the agent should avoid making

You do not need to fill in a fixed form. Natural language paragraphs, rough bullets, or lab notebook-style notes all work.

## Project Layout

```text
scRT-agent-v2/
  run_scrt_agent.py
  run_scrt_interactive.py
  run_scrt_figure.py
  environment.yml
  scrt_agent/
    agent.py
    hypothesis.py
    research.py
    validator.py
    notebook_tools.py
    figure_mode.py
    literature.py
    deepresearch.py
    interactive.py
    logger.py
    utils.py
    execution/
      legacy.py
    prompts/
      candidate_hypotheses.txt
      coding_system_prompt.txt
      coding_guidelines.txt
      first_draft.txt
      analysis_from_hypothesis.txt
      select_literature_hypothesis.txt
      critic.txt
      incorporate_critique.txt
      next_step.txt
      revise_hypothesis.txt
      interp_results.txt
      fix_code.txt
      deepresearch.txt
      step_research_update.txt
```

## Installation

```bash
conda env create -f environment.yml
conda activate scrt-agent-v2
```

Set:

```bash
OPENAI_API_KEY=...
```

Optional:

```bash
SCRT_VISION_MODEL=gpt-4o
SCRT_DEEP_RESEARCH_MODEL=o4-mini-deep-research
```

## Usage

```bash
python run_scrt_agent.py \
  --rna-h5ad-path PATH_TO_RNA_H5AD \
  --tcr-path PATH_TO_TCR_TABLE \
  --research-brief-path PATH_TO_BRIEF_TXT \
  --literature-path PATH_TO_PAPER_OR_FOLDER \
  --analysis-name MY_RUN \
  --with-figure \
  --num-analyses 3 \
  --max-iterations 6
```

Standalone figure export:

```bash
python run_scrt_figure.py \
  --rna-h5ad-path PATH_TO_RNA_H5AD \
  --tcr-path PATH_TO_TCR_TABLE \
  --output-dir FIGURE_OUTPUT_DIR
```

Interactive review flow:

```bash
python run_scrt_interactive.py prepare \
  --rna-h5ad-path PATH_TO_RNA_H5AD \
  --tcr-path PATH_TO_TCR_TABLE \
  --research-brief-path PATH_TO_BRIEF_TXT \
  --session-name MY_SESSION \
  --output-home SESSIONS_DIR

python run_scrt_interactive.py review \
  --session-dir SESSIONS_DIR/MY_SESSION \
  --candidate-index 2 \
  --feedback-text "Focus more on metastasis and avoid pooled claims."

python run_scrt_interactive.py run \
  --session-dir SESSIONS_DIR/MY_SESSION \
  --with-figure
```

## Design Notes

- The runtime assumes processed inputs and a Python-only environment.
- The default planning flow is `research brief first`; local literature is optional supporting material.
- It favors analyses that can be supported by merged RNA + TCR metadata inside `adata_rna.obs`.
- It explicitly warns against treating sample-local clonotype IDs as global clone IDs.
- It uses local literature not only as background text, but also to generate and rank candidate mechanisms before planning.
- It does not assume repertoire phylogeny, lineage trees, or advanced sequence models unless the TCR table already contains the needed annotations.
- It is still intentionally lightweight: validation is stronger than v1, but it is not yet a substitute for formal statistical review.
