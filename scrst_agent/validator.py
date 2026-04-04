"""Dataset validation for scRNA + spatial analysis."""

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
    def inspect_inputs(self, rna_h5ad_path: str | Path, spatial_h5ad_path: str | Path) -> DatasetValidationResult:
        result = DatasetValidationResult()
        rna = ad.read_h5ad(rna_h5ad_path, backed="r")
        spatial = ad.read_h5ad(spatial_h5ad_path, backed="r")
        try:
            shared_genes = len(set(map(str, rna.var_names)).intersection(map(str, spatial.var_names)))
            result.metrics = {
                "rna_cells": int(rna.n_obs),
                "rna_genes": int(rna.n_vars),
                "spatial_spots": int(spatial.n_obs),
                "spatial_genes": int(spatial.n_vars),
                "shared_genes": int(shared_genes),
            }
            if "X_umap" in rna.obsm:
                result.strengths.append("RNA object already contains a UMAP embedding.")
            if "X_umap" in spatial.obsm:
                result.strengths.append("Spatial object already contains a UMAP embedding.")
            if "spatial" in spatial.obsm:
                result.strengths.append("Spatial object contains spot coordinates.")
            else:
                result.warnings.append("Spatial object does not contain obsm['spatial']; figure panels may be limited.")
            for col in ("sample_id", "sample_key", "tissue", "cluster_cell_type"):
                if col in rna.obs.columns:
                    result.strengths.append(f"RNA metadata contains {col}.")
                    break
            if shared_genes >= 1000:
                result.strengths.append("RNA and spatial objects share enough genes for joint marker analysis.")
            else:
                result.warnings.append("RNA and spatial objects share few genes; marker-transfer analyses may be weak.")
            result.guardrails.extend(
                [
                    "Treat spot-level signals as mixed-cell signals unless deconvolution evidence is available.",
                    "Prefer tissue-aware or sample-aware comparisons over pooled global claims.",
                    "Do not over-interpret spatial correlation without checking shared tissue structure.",
                ]
            )
        finally:
            if getattr(rna, "file", None) is not None:
                rna.file.close()
            if getattr(spatial, "file", None) is not None:
                spatial.file.close()
        return result

