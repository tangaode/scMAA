"""Notebook executor for scRNA + spatial analysis."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell

from scrt_agent.execution.legacy import LegacyNotebookExecutor as BaseLegacyNotebookExecutor


class LegacyNotebookExecutor(BaseLegacyNotebookExecutor):
    """Notebook executor using a persistent Jupyter kernel for scRNA + spatial analysis."""

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
        spatial_summary: str,
        joint_summary: str,
        validation_summary: str,
        context_summary: str,
        logger,
        rna_h5ad_path: str,
        spatial_h5ad_path: str,
        spatial_mapping_method: str = "auto",
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
            tcr_summary=spatial_summary,
            joint_summary=joint_summary,
            validation_summary=validation_summary,
            context_summary=context_summary,
            logger=logger,
            rna_h5ad_path=rna_h5ad_path,
            tcr_path=spatial_h5ad_path,
            output_dir=output_dir,
            analysis_name=analysis_name,
            max_iterations=max_iterations,
            max_fix_attempts=max_fix_attempts,
            use_VLM=use_VLM,
            use_documentation=use_documentation,
        )
        self.spatial_summary = spatial_summary
        self.spatial_h5ad_path = str(Path(spatial_h5ad_path))
        self.spatial_mapping_method = spatial_mapping_method

    def create_initial_notebook(self, analysis, research_ledger) -> nbf.NotebookNode:
        notebook = nbf.v4.new_notebook()
        notebook.cells.append(new_markdown_cell("# scRNA + Spatial Analysis"))
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

from scrst_agent.notebook_tools import (
    cell2location_available,
    ensure_obs_column,
    ensure_obs_columns,
    expression_frame,
    get_spatial_mapping_method,
    infer_tumor_like_tissues,
    resolve_gene_names,
    run_cell2location_mapping,
    run_reference_mapping,
    safe_rank_genes_groups,
    spatial_coordinate_frame,
    spatial_domain_de,
    transfer_reference_markers_to_spatial,
    tumor_like_subset,
)

sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=100, facecolor="white", frameon=False)
plt.rcParams["figure.figsize"] = (8, 6)
sns.set_style("whitegrid")

RNA_H5AD_PATH = r'''{self.rna_h5ad_path}'''
SPATIAL_H5AD_PATH = r'''{self.spatial_h5ad_path}'''

print("Loading RNA and spatial inputs...")
adata_rna = sc.read_h5ad(RNA_H5AD_PATH)
adata_spatial = sc.read_h5ad(SPATIAL_H5AD_PATH)
ensure_obs_columns(adata_rna, ["sample_id", "sample_key", "tissue"], fill_value="Unknown", as_category=True)
ensure_obs_columns(adata_spatial, ["sample_id", "sample_key", "tissue"], fill_value="Unknown", as_category=True)
if "cell_type" not in adata_rna.obs.columns:
    for alias in ("cluster_cell_type", "celltype", "celltypes", "annotation", "annotated_cell_type"):
        if alias in adata_rna.obs.columns:
            adata_rna.obs["cell_type"] = adata_rna.obs[alias].astype(str)
            print(f"Created RNA cell_type alias from {{alias}}")
            break
ensure_obs_column(adata_rna, "cell_type", fill_value="Unknown", as_category=True)
if "spatial_domain" not in adata_spatial.obs.columns:
    for alias in ("cluster_cell_type", "leiden", "domain"):
        if alias in adata_spatial.obs.columns:
            adata_spatial.obs["spatial_domain"] = adata_spatial.obs[alias].astype(str)
            print(f"Created spatial_domain alias from {{alias}}")
            break
ensure_obs_column(adata_spatial, "spatial_domain", fill_value="Unknown", as_category=True)
shared_genes = sorted(set(map(str, adata_rna.var_names)).intersection(map(str, adata_spatial.var_names)))
print(f"RNA shape: {{adata_rna.n_obs}} x {{adata_rna.n_vars}}")
print(f"Spatial shape: {{adata_spatial.n_obs}} x {{adata_spatial.n_vars}}")
print(f"Shared genes: {{len(shared_genes)}}")
SPATIAL_MAPPING_METHOD = get_spatial_mapping_method(r'''{self.spatial_mapping_method}''')
print(f"Spatial mapping method: {{SPATIAL_MAPPING_METHOD}}")
print(f"cell2location available: {{cell2location_available()}}")
print("Notebook helper functions available: tumor_like_subset, resolve_gene_names, expression_frame, spatial_coordinate_frame, transfer_reference_markers_to_spatial, run_cell2location_mapping, run_reference_mapping, spatial_domain_de")
"""
        notebook.cells.append(new_code_cell(setup_code))
        return notebook
