"""Notebook executor for scRNA + scATAC analysis."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell

from scrt_agent.execution.legacy import LegacyNotebookExecutor as BaseLegacyNotebookExecutor


class LegacyNotebookExecutor(BaseLegacyNotebookExecutor):
    """Notebook executor using a persistent Jupyter kernel for scRNA + scATAC analysis."""

    def __init__(
        self,
        *,
        hypothesis_generator,
        openai_api_key: str,
        model_name: str,
        vision_model: str,
        prompt_dir: str | Path,
        coding_guidelines: str,
        coding_system_prompt: str,
        rna_summary: str,
        atac_summary: str,
        joint_summary: str,
        validation_summary: str,
        context_summary: str,
        logger,
        rna_h5ad_path: str,
        atac_h5ad_path: str,
        output_dir: str | Path,
        analysis_name: str,
        max_iterations: int = 6,
        max_fix_attempts: int = 3,
        use_VLM: bool = True,
        use_documentation: bool = True,
    ) -> None:
        super().__init__(
            hypothesis_generator=hypothesis_generator,
            openai_api_key=openai_api_key,
            model_name=model_name,
            vision_model=vision_model,
            prompt_dir=prompt_dir,
            coding_guidelines=coding_guidelines,
            coding_system_prompt=coding_system_prompt,
            rna_summary=rna_summary,
            tcr_summary=atac_summary,
            joint_summary=joint_summary,
            validation_summary=validation_summary,
            context_summary=context_summary,
            logger=logger,
            rna_h5ad_path=rna_h5ad_path,
            tcr_path=atac_h5ad_path,
            output_dir=output_dir,
            analysis_name=analysis_name,
            max_iterations=max_iterations,
            max_fix_attempts=max_fix_attempts,
            use_VLM=use_VLM,
            use_documentation=use_documentation,
        )
        self.atac_summary = atac_summary
        self.atac_h5ad_path = str(Path(atac_h5ad_path))

    def create_initial_notebook(self, analysis, research_ledger) -> nbf.NotebookNode:
        notebook = nbf.v4.new_notebook()
        notebook.cells.append(new_markdown_cell("# scRNA + scATAC Analysis"))
        notebook.cells.append(new_markdown_cell(f"## Hypothesis\n\n{analysis.hypothesis}"))
        notebook.cells.append(
            new_markdown_cell(
                "## Research Framing\n\n"
                f"Priority question: {analysis.priority_question}\n\n"
                f"Evidence goal: {analysis.evidence_goal}\n\n"
                f"Decision rationale: {analysis.decision_rationale}\n\n"
                "Validation checks:\n" + "\n".join(f"- {item}" for item in analysis.validation_checks)
            )
        )
        notebook.cells.append(new_markdown_cell(f"## Dataset Validation\n\n{self.validation_summary}"))
        notebook.cells.append(new_markdown_cell(f"## Initial Research Ledger\n\n{research_ledger.to_markdown()}"))

        setup_code = f"""import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

PROJECT_ROOT = r'''{self.project_root}'''
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scrat_agent.notebook_tools import (
    ensure_obs_column,
    ensure_obs_columns,
    infer_tumor_like_tissues,
    tumor_like_subset,
    resolve_gene_names,
    expression_frame,
    atac_signal_frame,
    find_shared_obs_names,
    paired_modality_subset,
    safe_rank_genes_groups,
    safe_rank_features_groups,
    summarize_rna_atac_link,
)

sc.settings.verbosity = 2
try:
    from IPython import get_ipython
    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic("matplotlib", "inline")
except Exception:
    pass
try:
    plt.switch_backend("module://matplotlib_inline.backend_inline")
except Exception:
    pass
sc.settings.set_figure_params(dpi=100, facecolor="white", frameon=False)
plt.rcParams["figure.figsize"] = (8, 6)
sns.set_style("whitegrid")

RNA_H5AD_PATH = r'''{self.rna_h5ad_path}'''
ATAC_H5AD_PATH = r'''{self.atac_h5ad_path}'''

print("Loading RNA and ATAC inputs...")
adata_rna = sc.read_h5ad(RNA_H5AD_PATH)
adata_atac = sc.read_h5ad(ATAC_H5AD_PATH)
ensure_obs_columns(adata_rna, ["sample_id", "sample_key", "tissue"], fill_value="Unknown", as_category=True)
ensure_obs_columns(adata_atac, ["sample_id", "sample_key", "tissue"], fill_value="Unknown", as_category=True)
if "cell_type" not in adata_rna.obs.columns:
    for alias in ("cluster_cell_type", "celltype", "annotation"):
        if alias in adata_rna.obs.columns:
            adata_rna.obs["cell_type"] = adata_rna.obs[alias].astype(str)
            print(f"Created RNA cell_type alias from {{alias}}")
            break
ensure_obs_column(adata_rna, "cell_type", fill_value="Unknown", as_category=True)
if "cell_type" not in adata_atac.obs.columns:
    for alias in ("cluster_cell_type", "celltype", "annotation", "leiden"):
        if alias in adata_atac.obs.columns:
            adata_atac.obs["cell_type"] = adata_atac.obs[alias].astype(str)
            print(f"Created ATAC cell_type alias from {{alias}}")
            break
ensure_obs_column(adata_atac, "cell_type", fill_value="Unknown", as_category=True)
shared_obs_names = find_shared_obs_names(adata_rna, adata_atac)
shared_gene_features = sorted(set(map(str, adata_rna.var_names)).intersection(map(str, adata_atac.var_names)))
print(f"RNA shape: {{adata_rna.n_obs}} x {{adata_rna.n_vars}}")
print(f"ATAC shape: {{adata_atac.n_obs}} x {{adata_atac.n_vars}}")
print(f"Shared obs names: {{len(shared_obs_names)}}")
print(f"Shared gene-like features: {{len(shared_gene_features)}}")
print("Notebook helper functions available: tumor_like_subset, resolve_gene_names, expression_frame, atac_signal_frame, paired_modality_subset, safe_rank_features_groups, summarize_rna_atac_link")
"""
        notebook.cells.append(new_code_cell(setup_code))
        return notebook
