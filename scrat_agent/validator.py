"""Dataset validation for scRNA + scATAC analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad


@dataclass
class DatasetValidationResult:
    metrics: dict[str, object] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        lines = ["Metrics:"]
        lines.extend(f"- {key}: {value}" for key, value in self.metrics.items())
        lines.append("Strengths:")
        lines.extend(f"- {item}" for item in self.strengths)
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in self.warnings)
        lines.append("Guardrails:")
        lines.extend(f"- {item}" for item in self.guardrails)
        return "\n".join(lines)


class DatasetValidator:
    def inspect_inputs(self, rna_h5ad_path: str | Path, atac_h5ad_path: str | Path) -> DatasetValidationResult:
        result = DatasetValidationResult()
        rna = ad.read_h5ad(rna_h5ad_path, backed="r")
        atac = ad.read_h5ad(atac_h5ad_path, backed="r")
        try:
            shared_obs = len(set(map(str, rna.obs_names)).intersection(map(str, atac.obs_names)))
            shared_genes = len(set(map(str, rna.var_names)).intersection(map(str, atac.var_names)))
            result.metrics = {
                "rna_cells": int(rna.n_obs),
                "rna_genes": int(rna.n_vars),
                "atac_cells": int(atac.n_obs),
                "atac_features": int(atac.n_vars),
                "shared_obs_names": int(shared_obs),
                "shared_gene_like_features": int(shared_genes),
            }
            if "X_umap" in rna.obsm:
                result.strengths.append("RNA object already contains a UMAP embedding.")
            if "X_umap" in atac.obsm:
                result.strengths.append("ATAC object already contains a UMAP embedding.")
            if "gene_activity" in atac.layers:
                result.strengths.append("ATAC object contains a gene_activity layer.")
            elif "gene_activity" in atac.obsm and "gene_activity_var_names" in atac.uns:
                result.strengths.append("ATAC object contains gene_activity signals in obsm.")
            if "gene_name" in atac.var.columns:
                result.strengths.append("ATAC object contains gene_name annotations for features.")
            if "gene_activity_var_names" in atac.uns:
                result.metrics["gene_activity_features"] = int(len(atac.uns["gene_activity_var_names"]))
            if shared_obs >= 100:
                result.strengths.append("RNA and ATAC objects share many observation names, enabling paired analyses.")
            else:
                result.warnings.append("RNA and ATAC objects share few observation names; paired multiome analyses may be limited.")
            if shared_genes >= 500:
                result.strengths.append("ATAC feature annotations overlap with RNA genes enough for gene-level comparisons.")
            else:
                result.warnings.append("Direct gene-level overlap between RNA and ATAC is limited; rely on gene activity or annotated features.")
            result.guardrails.extend(
                [
                    "Do not treat chromatin accessibility as equivalent to gene expression.",
                    "If gene_activity is unavailable, state that peak-level signals are only indirect proxies for transcription.",
                    "Prefer paired-cell, paired-sample, or shared-group comparisons over pooled global claims.",
                    "Avoid causal claims about regulation unless both RNA and ATAC evidence point in the same direction.",
                ]
            )
        finally:
            if getattr(rna, "file", None) is not None:
                rna.file.close()
            if getattr(atac, "file", None) is not None:
                atac.file.close()
        return result
