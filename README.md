# scMAA

`scMAA` stands for `scMultiomic-Analysis-Agent`.

This project supports three workflows in one desktop app:

- `scRNA + scTCR`
- `scRNA + spatial transcriptomics`
- `scRNA + scATAC`

The desktop app is the main entry point. Users can choose the workflow from a drop-down menu before preparing data or running the agent.

## Inputs

For the main agent, provide:

- one processed RNA file in `.h5ad` format
- one second modality file
- one text file that explains the research question

The second modality depends on the selected workflow:

- `scRNA + scTCR`: one TCR table such as `.tsv`, `.csv`, or `.txt`
- `scRNA + spatial transcriptomics`: one processed spatial `.h5ad`
- `scRNA + scATAC`: one processed ATAC `.h5ad`

Optional:

- local papers, reviews, or notes

The main text input is the `research brief`. It does not need a strict template. A few short paragraphs or bullet points are enough.

## Raw Data Preparation

### scRNA + scTCR

If the raw files look like:

- `*_barcodes.tsv.gz`
- `*_features.tsv.gz`
- `*_matrix.mtx.gz`
- `*_filtered_contig_annotations.csv.gz`
- or one `GSE*_RAW.tar`

run:

```bash
python run_scrt_prepare_data.py \
  --raw-input-path PATH_TO_RAW_DIR_OR_RAW_TAR \
  --output-dir PREPARED_OUTPUT_DIR
```

### scRNA + spatial transcriptomics

Run:

```bash
python run_scrst_prepare_data.py \
  --rna-raw-input-path PATH_TO_RNA_RAW_DIR_OR_TAR \
  --spatial-raw-input-path PATH_TO_SPATIAL_RAW_DIR_OR_TAR \
  --output-dir PREPARED_OUTPUT_DIR
```

### scRNA + scATAC

For common `10x multiome / cellranger-arc` raw folders or `RAW.tar`, run:

```bash
python run_scrat_prepare_data.py \
  --raw-input-path PATH_TO_RAW_DIR_OR_RAW_TAR \
  --output-dir PREPARED_OUTPUT_DIR
```

This workflow will try the built-in parser first. If the raw format is more complex, the agent can fall back to model-generated preprocessing code and continue automatically.

## Install

```bash
conda env create -f environment.yml
conda activate scmaa
```

Set your API key:

```bash
OPENAI_API_KEY=...
```

If you also use DeepSeek, place `OPENAI.env` and `deepseek.env` in the project root.

## Desktop App

Double-click:

- `launch_scMAA_gui.bat`

or run:

```bash
python run_scmaa_gui.pyw
```

The desktop app lets you:

- choose `scRNA-TCR`, `scRNA-ST`, or `scRNA-ATAC`
- prepare raw data into processed input files
- generate or regenerate candidate hypotheses
- regenerate an analysis plan from user feedback
- approve one hypothesis and one plan
- run the analysis
- save standard summary figures and hypothesis-driven figures

## Command Line

Mode-specific CLIs are still available for advanced use:

- `run_scrt_agent.py`
- `run_scrst_agent.py`
- `run_scrat_agent.py`

Example for `scRNA + scTCR`:

```bash
python run_scrt_agent.py \
  --rna-h5ad-path PREPARED_OUTPUT_DIR/processed_rna.h5ad \
  --tcr-path PREPARED_OUTPUT_DIR/merged_tcr_annotations.tsv.gz \
  --research-brief-path PATH_TO_BRIEF_TXT \
  --analysis-name MY_RUN \
  --with-figure
```

Example for `scRNA + spatial transcriptomics`:

```bash
python run_scrst_agent.py \
  --rna-h5ad-path PREPARED_OUTPUT_DIR/processed_rna.h5ad \
  --spatial-h5ad-path PREPARED_OUTPUT_DIR/processed_spatial.h5ad \
  --research-brief-path PATH_TO_BRIEF_TXT \
  --analysis-name MY_RUN \
  --with-figure
```

Example for `scRNA + scATAC`:

```bash
python run_scrat_agent.py \
  --rna-h5ad-path PREPARED_OUTPUT_DIR/processed_rna.h5ad \
  --atac-h5ad-path PREPARED_OUTPUT_DIR/processed_atac.h5ad \
  --research-brief-path PATH_TO_BRIEF_TXT \
  --analysis-name MY_RUN \
  --with-figure
```

## Main Outputs

- `run_summary.txt`
- `*_analysis_1.ipynb`
- `figure/*_publication_figure.png`
- `figure/*_publication_figure_hypothesis.png`
- `figure/*.pdf`
- `logs/`

If you only want to inspect one file first, open `run_summary.txt`.

## Notes

- `scMAA` is the product name. Internal package names are still kept as-is for compatibility.
- The research brief is the main input. Local papers are optional.
- Results still need user review before they are treated as biological conclusions.
- The desktop app is the easiest way to start.
