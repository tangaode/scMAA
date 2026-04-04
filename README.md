# scMAA

`scMAA` stands for `scMultiomic-Analysis-Agent`.

It is a desktop-first tool for three single-cell workflows:

- `scRNA + scTCR`
- `scRNA + spatial transcriptomics`
- `scRNA + scATAC`

Users can prepare raw data, review candidate hypotheses, adjust the analysis plan, run the notebook workflow, and export both standard summary figures and hypothesis-driven figures.

## What It Does

- prepares common raw inputs into processed `.h5ad` files
- generates candidate hypotheses from the research brief
- lets the user revise the plan before execution
- runs notebook-based analysis
- exports `run_summary.txt`, notebook outputs, and figures

## Main Entry

The easiest way to start is the desktop app:

- `launch_scMAA_gui.bat`

or:

```bash
python run_scmaa_gui.pyw
```

Inside the app, choose one mode:

- `scRNA-TCR`
- `scRNA-ST`
- `scRNA-ATAC`

## Inputs

For the main analysis step, provide:

- one processed RNA `.h5ad`
- one second modality file
- one `research brief` text file

The second modality file depends on the mode:

- `scRNA + scTCR`: one TCR table such as `.tsv`, `.csv`, or `.txt`
- `scRNA + spatial transcriptomics`: one processed spatial `.h5ad`
- `scRNA + scATAC`: one processed ATAC `.h5ad`

Optional:

- local papers, reviews, or notes

The `research brief` can be short. A few paragraphs or bullet points are enough.

## Raw Preparation

### scRNA + scTCR

```bash
python run_scrt_prepare_data.py \
  --raw-input-path PATH_TO_RAW_DIR_OR_RAW_TAR \
  --output-dir PREPARED_OUTPUT_DIR
```

### scRNA + spatial transcriptomics

```bash
python run_scrst_prepare_data.py \
  --rna-raw-input-path PATH_TO_RNA_RAW_DIR_OR_TAR \
  --spatial-raw-input-path PATH_TO_SPATIAL_RAW_DIR_OR_TAR \
  --output-dir PREPARED_OUTPUT_DIR
```

### scRNA + scATAC

```bash
python run_scrat_prepare_data.py \
  --raw-input-path PATH_TO_RAW_DIR_OR_RAW_TAR \
  --output-dir PREPARED_OUTPUT_DIR
```

For common `10x multiome / cellranger-arc` data, the built-in parser is used first. For more complex raw formats, the preprocessing step can fall back to model-generated code and continue automatically.

## Install

```bash
conda env create -f environment.yml
conda activate scmaa
```

Set your API key:

```bash
OPENAI_API_KEY=...
```

You can also place `OPENAI.env` and `deepseek.env` in the project root.

## Command Line

Mode-specific CLIs are also available:

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

## Outputs

Typical outputs:

- `run_summary.txt`
- `*_analysis_1.ipynb`
- `figure/*_publication_figure.png`
- `figure/*_publication_figure_hypothesis.png`
- `figure/*.pdf`
- `logs/`

If you only open one file first, open `run_summary.txt`.

## Notes

- `scMAA` is the product name. Internal package names are kept as they are for compatibility.
- The research brief is the main input. Papers are optional.
- Results still need user review before they are treated as biological conclusions.
