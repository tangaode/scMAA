# scRT-agent v2

`scRT-agent v2` is a tool for `scRNA + scTCR` analysis.

It can start from raw GEO-style files or from a prepared RNA object and TCR table.

If you start from raw data, it first builds a processed `.h5ad`, merges the TCR tables, runs basic QC, UMAP, clustering, cluster markers, and cluster labels. Then the main agent can use those prepared files.

## Inputs

For the main agent:

- one processed RNA file in `.h5ad` format
- one scTCR table such as `.tsv`, `.csv`, or `.txt`
- one text file that explains what you want to study

Optional:

- local papers, reviews, or notes

The main text input is the `research brief`. It does not need a fixed template. A few short paragraphs or bullet points are enough. It helps to mention the question, the samples or tissues, the main comparison, and any markers or pathways you care about.

## Raw Data Preparation

If your data is still in raw files such as:

- `*_barcodes.tsv.gz`
- `*_features.tsv.gz`
- `*_matrix.mtx.gz`
- `*_filtered_contig_annotations.csv.gz`
- or one `GSE*_RAW.tar`

run this first:

```bash
python run_scrt_prepare_data.py \
  --raw-input-path PATH_TO_RAW_DIR_OR_RAW_TAR \
  --output-dir PREPARED_OUTPUT_DIR
```

This step writes:

- `processed_rna.h5ad`
- `merged_tcr_annotations.tsv.gz`
- `cluster_markers.csv`
- `cluster_annotations.csv`
- `qc_summary.txt`
- `figures/umap_leiden.png`
- `figures/umap_cluster_cell_type.png`

The preparation step also:

- filters low-quality cells
- computes PCA, neighbors, UMAP, and Leiden clusters
- exports cluster marker genes
- sends the top 50 non-lincRNA markers of each cluster to the model for cluster labeling

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
  --rna-h5ad-path PREPARED_OUTPUT_DIR/processed_rna.h5ad \
  --tcr-path PREPARED_OUTPUT_DIR/merged_tcr_annotations.tsv.gz \
  --research-brief-path PATH_TO_BRIEF_TXT \
  --analysis-name MY_RUN \
  --with-figure
```

If you want to add local papers or notes:

```bash
python run_scrt_agent.py \
  --rna-h5ad-path PREPARED_OUTPUT_DIR/processed_rna.h5ad \
  --tcr-path PREPARED_OUTPUT_DIR/merged_tcr_annotations.tsv.gz \
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
