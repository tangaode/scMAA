# scRT-agent v2

`scRT-agent v2` is a tool for `scRNA + scTCR` analysis.

It reads a processed RNA file, a TCR table, and a short research note from the user. Then it proposes a question, writes notebook code, runs the code, and saves the results.

## What It Can Do

- read a processed `.h5ad` RNA object
- read a TCR table such as `.tsv`, `.csv`, or `.txt`
- use a freeform research note as the main input
- optionally read local papers or notes as extra background
- suggest candidate hypotheses before running analysis
- let the user pick or revise a hypothesis
- run notebook-based analysis
- export a summary figure

## What You Need

- one processed scRNA file in `.h5ad` format
- one scTCR table
- one text file that explains what you want to study

Optional:

- local papers, reviews, or notes

## About The Research Brief

The main text input is a `research brief`.

You do not need to fill in a form. Write it in your own way. A few short paragraphs or bullet points are enough.

Useful things to mention:

- what question you care about
- what the tissues or samples mean
- what comparison matters most
- any markers or pathways you care about
- what kind of result would be useful
- anything the tool should avoid claiming

## Install

```bash
conda env create -f environment.yml
conda activate scrt-agent-v2
```

Set your API key:

```bash
OPENAI_API_KEY=...
```

## Main Run

```bash
python run_scrt_agent.py \
  --rna-h5ad-path PATH_TO_RNA_H5AD \
  --tcr-path PATH_TO_TCR_TABLE \
  --research-brief-path PATH_TO_BRIEF_TXT \
  --analysis-name MY_RUN \
  --with-figure
```

If you have local papers or notes, add them like this:

```bash
python run_scrt_agent.py \
  --rna-h5ad-path PATH_TO_RNA_H5AD \
  --tcr-path PATH_TO_TCR_TABLE \
  --research-brief-path PATH_TO_BRIEF_TXT \
  --literature-path PATH_TO_PAPER_OR_FOLDER \
  --analysis-name MY_RUN \
  --with-figure
```

## Interactive Use

If you want to review the hypothesis before the notebook runs, use the interactive mode.

### Step 1: Prepare

```bash
python run_scrt_interactive.py prepare \
  --rna-h5ad-path PATH_TO_RNA_H5AD \
  --tcr-path PATH_TO_TCR_TABLE \
  --research-brief-path PATH_TO_BRIEF_TXT \
  --session-name MY_SESSION \
  --output-home SESSIONS_DIR
```

This step creates:

- `candidate_hypotheses.json`
- `candidate_hypotheses.md`

### Step 2: Review

Pick one candidate or give your own feedback.

```bash
python run_scrt_interactive.py review \
  --session-dir SESSIONS_DIR/MY_SESSION \
  --candidate-index 2 \
  --feedback-text "Focus more on metastasis and avoid pooled claims."
```

This step creates:

- `approved_hypothesis.txt`
- `approved_plan.json`
- `approved_plan.md`

### Step 3: Run

```bash
python run_scrt_interactive.py run \
  --session-dir SESSIONS_DIR/MY_SESSION \
  --with-figure
```

## Figure Export

If you only want the summary figure, run:

```bash
python run_scrt_figure.py \
  --rna-h5ad-path PATH_TO_RNA_H5AD \
  --tcr-path PATH_TO_TCR_TABLE \
  --output-dir FIGURE_OUTPUT_DIR
```

## Output Files

The main run usually creates these files:

- `run_summary.txt`
- `*_analysis_1.ipynb`
- `figure/*.png`
- `figure/*.pdf`
- `logs/`

If you only want to look at one file first, open `run_summary.txt`.

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
    interactive.py
    figure_mode.py
    notebook_tools.py
    validator.py
    literature.py
    execution/
      legacy.py
```

## Notes

- This project expects processed input files. It does not start from raw sequencing files.
- The main input is the research brief. Local papers are optional.
- The code tries to avoid unsafe clone-level claims when `clonotype_id` looks sample-local.
- The results still need user review. Do not treat them as final biological conclusions without checking them yourself.
