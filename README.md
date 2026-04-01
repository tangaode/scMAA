# scRT-agent v2

`scRT-agent v2` is a tool for `scRNA + scTCR` analysis.

It takes a processed RNA file, a TCR table, and a short research brief. It then proposes a question, writes notebook code, runs the code, and saves the results.

## Inputs

Required:

- one processed RNA file in `.h5ad` format
- one scTCR table such as `.tsv`, `.csv`, or `.txt`
- one text file that explains what you want to study

Optional:

- local papers, reviews, or notes

The main text input is the `research brief`. It does not need a fixed template. A few short paragraphs or bullet points are enough. It helps to mention the question, the samples or tissues, the main comparison, and any markers or pathways you care about.

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

If you want to add local papers or notes:

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

Use the interactive mode if you want to review the hypothesis before the notebook runs.

Prepare candidates:

```bash
python run_scrt_interactive.py prepare \
  --rna-h5ad-path PATH_TO_RNA_H5AD \
  --tcr-path PATH_TO_TCR_TABLE \
  --research-brief-path PATH_TO_BRIEF_TXT \
  --session-name MY_SESSION \
  --output-home SESSIONS_DIR
```

Review and revise:

```bash
python run_scrt_interactive.py review \
  --session-dir SESSIONS_DIR/MY_SESSION \
  --candidate-index 2 \
  --feedback-text "Focus more on metastasis and avoid pooled claims."
```

Run the approved plan:

```bash
python run_scrt_interactive.py run \
  --session-dir SESSIONS_DIR/MY_SESSION \
  --with-figure
```

## Figure Only

```bash
python run_scrt_figure.py \
  --rna-h5ad-path PATH_TO_RNA_H5AD \
  --tcr-path PATH_TO_TCR_TABLE \
  --output-dir FIGURE_OUTPUT_DIR
```

## Main Outputs

- `run_summary.txt`
- `*_analysis_1.ipynb`
- `figure/*.png`
- `figure/*.pdf`
- `logs/`

If you only want to check one file first, open `run_summary.txt`.

## Notes

- This project expects processed input files, not raw sequencing files.
- The research brief is the main input. Local papers are optional.
- Results still need user review before they are treated as biological conclusions.
