"""Reusable notebook helper functions for scMAA."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import scanpy as sc


TUMOR_HINT_TOKENS = ("tumor", "metast", "primary", "focus", "lesion", "cancer", "carcinoma", "malignan")
NON_TUMOR_HINT_TOKENS = ("pbmc", "blood", "normal", "healthy", "control", "adjacent", "benign", "spleen")
COMMON_GENE_ALIASES = {
    "PD1": "PDCD1",
    "PD-1": "PDCD1",
    "PDCD1": "PDCD1",
    "TIGIT": "TIGIT",
    "TIM3": "HAVCR2",
    "TIM-3": "HAVCR2",
    "LAG3": "LAG3",
    "CTLA4": "CTLA4",
    "XBP-1": "XBP1",
    "XBP1": "XBP1",
}


def ensure_obs_column(adata, column: str, fill_value: str = "Unknown", as_category: bool = True) -> None:
    """Ensure an obs column exists and is optionally categorical."""
    if column not in adata.obs.columns:
        adata.obs[column] = fill_value
    else:
        adata.obs[column] = adata.obs[column].astype("object").where(adata.obs[column].notna(), fill_value)
    if as_category:
        adata.obs[column] = adata.obs[column].astype("category")


def ensure_obs_columns(adata, columns: Iterable[str], fill_value: str = "Unknown", as_category: bool = True) -> None:
    for column in columns:
        ensure_obs_column(adata, column, fill_value=fill_value, as_category=as_category)


def paired_tcr_subset(adata, copy: bool = True):
    """Return the paired scRNA + scTCR subset."""
    if "has_tcr" not in adata.obs.columns:
        raise KeyError("'has_tcr' is not present in adata.obs.")
    mask = adata.obs["has_tcr"].fillna(False).astype(bool)
    return adata[mask].copy() if copy else adata[mask]


def infer_tumor_like_tissues(adata, tissue_col: str = "tissue") -> list[str]:
    """Infer tumor-like tissue labels from observed metadata values."""
    if tissue_col not in adata.obs.columns:
        raise KeyError(f"'{tissue_col}' is not present in adata.obs.")
    labels = (
        adata.obs[tissue_col]
        .dropna()
        .astype(str)
        .map(str.strip)
    )
    inferred: list[str] = []
    for label in labels.value_counts(dropna=False).index.tolist():
        lowered = label.lower()
        if any(token in lowered for token in NON_TUMOR_HINT_TOKENS):
            continue
        if any(token in lowered for token in TUMOR_HINT_TOKENS):
            inferred.append(label)
    return inferred


def tumor_like_subset(adata, tissue_col: str = "tissue", copy: bool = True):
    """Return a tumor-like subset using heuristic tissue label inference."""
    tissues = infer_tumor_like_tissues(adata, tissue_col=tissue_col)
    if not tissues:
        raise ValueError(
            f"No tumor-like labels inferred from '{tissue_col}'. "
            "Inspect the available tissue values before making tumor-specific claims."
        )
    mask = adata.obs[tissue_col].astype(str).isin(tissues)
    subset = adata[mask].copy() if copy else adata[mask]
    print(f"Tumor-like tissues inferred from {tissue_col}: {', '.join(tissues)}")
    print(f"Tumor-like subset cells: {subset.n_obs}")
    return subset


def resolve_gene_names(adata, genes: Iterable[str]) -> dict[str, str]:
    """Resolve requested markers to canonical dataset gene names."""
    var_lookup = {str(name).upper(): str(name) for name in adata.var_names}
    resolved: dict[str, str] = {}
    for gene in genes:
        requested = str(gene).strip()
        if not requested:
            continue
        candidates = [requested]
        alias = COMMON_GENE_ALIASES.get(requested.upper())
        if alias and alias not in candidates:
            candidates.append(alias)
        if requested.upper() in COMMON_GENE_ALIASES:
            canonical = COMMON_GENE_ALIASES[requested.upper()]
            if canonical not in candidates:
                candidates.append(canonical)
        matched = None
        for candidate in candidates:
            matched = var_lookup.get(candidate.upper())
            if matched:
                break
        if matched:
            resolved[requested] = matched
    return resolved


def expression_frame(adata, genes: Iterable[str], obs_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Extract a dense expression DataFrame using resolved marker names."""
    resolved = resolve_gene_names(adata, genes)
    actual_genes = list(dict.fromkeys(resolved.values()))
    if not actual_genes:
        raise ValueError("None of the requested genes were found in adata.var_names after alias resolution.")
    matrix = adata[:, actual_genes].X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    frame = pd.DataFrame(matrix, index=adata.obs_names, columns=actual_genes)
    if obs_columns:
        for column in obs_columns:
            if column in adata.obs.columns:
                frame[column] = adata.obs[column].values
    print("Resolved genes:", ", ".join(f"{src}->{dst}" for src, dst in resolved.items()))
    return frame


def clone_expansion_table(adata, groupby: str = "tissue", paired_only: bool = True) -> pd.DataFrame:
    """Summarize expanded-clone counts by grouping column."""
    if paired_only:
        adata = paired_tcr_subset(adata, copy=True)
    if "expanded_clone" not in adata.obs.columns:
        raise KeyError("'expanded_clone' is not present in adata.obs.")
    ensure_obs_column(adata, groupby, fill_value="Unknown", as_category=True)
    summary = (
        adata.obs.groupby(groupby, observed=False)["expanded_clone"]
        .agg(total_paired="size", expanded_cells="sum")
        .reset_index()
    )
    summary["expanded_fraction"] = summary["expanded_cells"] / summary["total_paired"].clip(lower=1)
    return summary.sort_values(["expanded_fraction", "expanded_cells"], ascending=[False, False]).reset_index(drop=True)


def print_clone_expansion_table(adata, groupby: str = "tissue", paired_only: bool = True) -> pd.DataFrame:
    """Print and return the clone expansion summary."""
    summary = clone_expansion_table(adata, groupby=groupby, paired_only=paired_only)
    print(summary.to_string(index=False))
    return summary


def safe_rank_genes_groups(
    adata,
    *,
    groupby: str,
    groups: list[str] | None = None,
    reference: str | None = None,
    method: str = "wilcoxon",
    layer: str | None = None,
    use_raw: bool | None = None,
    min_cells_per_group: int = 20,
    key_added: str | None = None,
):
    """Run rank_genes_groups after enforcing categorical group labels and minimum group sizes."""
    ensure_obs_column(adata, groupby, fill_value="Unknown", as_category=True)
    counts = adata.obs[groupby].value_counts(dropna=False)
    eligible = counts[counts >= min_cells_per_group].index.astype(str).tolist()
    if len(eligible) < 2:
        raise ValueError(
            f"Not enough groups with >= {min_cells_per_group} cells for rank_genes_groups on '{groupby}'."
        )

    working = adata[adata.obs[groupby].astype(str).isin(eligible)].copy()
    working.obs[groupby] = working.obs[groupby].astype(str).astype("category")
    selected_groups = groups
    if selected_groups is not None:
        selected_groups = [item for item in groups if item in set(eligible)]
        if not selected_groups:
            raise ValueError(f"Requested groups are not eligible for '{groupby}'.")

    sc.tl.rank_genes_groups(
        working,
        groupby=groupby,
        groups=selected_groups,
        reference=reference,
        method=method,
        layer=layer,
        use_raw=use_raw,
        key_added=key_added,
    )
    return working


def tissue_stratified_expansion_de(
    adata,
    *,
    tissue_col: str = "tissue",
    expansion_col: str = "expanded_clone",
    paired_only: bool = True,
    min_cells_per_group: int = 20,
    top_n: int = 10,
    method: str = "wilcoxon",
    **kwargs,
) -> pd.DataFrame:
    """Run expanded-vs-non-expanded DE within each tissue using safe categorical handling."""
    tissue_col = kwargs.pop("tissue", tissue_col)
    expansion_col = kwargs.pop("expanded", expansion_col)
    expansion_col = kwargs.pop("group_col", expansion_col)
    expansion_col = kwargs.pop("expansion", expansion_col)
    kwargs.pop("sample_aware", None)
    if kwargs:
        print(f"Ignoring unsupported kwargs in tissue_stratified_expansion_de: {sorted(kwargs)}")
    if paired_only:
        adata = paired_tcr_subset(adata, copy=True)
    ensure_obs_columns(adata, [tissue_col], fill_value="Unknown", as_category=True)
    if expansion_col not in adata.obs.columns:
        raise KeyError(f"'{expansion_col}' is not present in adata.obs.")

    adata.obs[expansion_col] = adata.obs[expansion_col].fillna(False).astype(bool).map(
        {True: "expanded", False: "non_expanded"}
    )
    adata.obs[expansion_col] = adata.obs[expansion_col].astype("category")

    frames: list[pd.DataFrame] = []
    for tissue in adata.obs[tissue_col].astype(str).value_counts(dropna=False).index.tolist():
        tissue_data = adata[adata.obs[tissue_col].astype(str) == tissue].copy()
        counts = tissue_data.obs[expansion_col].value_counts()
        expanded_n = int(counts.get("expanded", 0))
        non_expanded_n = int(counts.get("non_expanded", 0))
        print(
            f"Tissue={tissue}: expanded={expanded_n}, non_expanded={non_expanded_n}, total={tissue_data.n_obs}"
        )
        if expanded_n < min_cells_per_group or non_expanded_n < min_cells_per_group:
            print(
                f"Skipping tissue={tissue} because one group has fewer than {min_cells_per_group} cells."
            )
            continue

        ranked = safe_rank_genes_groups(
            tissue_data,
            groupby=expansion_col,
            groups=["expanded"],
            reference="non_expanded",
            method=method,
            min_cells_per_group=min_cells_per_group,
            key_added=f"{tissue}_expanded_vs_non_expanded",
        )
        frame = sc.get.rank_genes_groups_df(
            ranked,
            group="expanded",
            key=f"{tissue}_expanded_vs_non_expanded",
        ).head(top_n)
        frame.insert(0, "tissue", tissue)
        frame.insert(1, "expanded_n", expanded_n)
        frame.insert(2, "non_expanded_n", non_expanded_n)
        frames.append(frame)

    if not frames:
        raise ValueError("No tissues had enough cells for stratified differential expression.")
    result = pd.concat(frames, ignore_index=True)
    print(result[["tissue", "names", "scores", "logfoldchanges", "pvals_adj"]].to_string(index=False))
    return result

