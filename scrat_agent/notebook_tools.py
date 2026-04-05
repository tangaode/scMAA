"""Reusable notebook helper functions for scRNA + scATAC analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import scanpy as sc


TUMOR_HINT_TOKENS = ("tumor", "metast", "primary", "focus", "lesion", "cancer", "carcinoma", "malignan")
NON_TUMOR_HINT_TOKENS = ("pbmc", "blood", "normal", "healthy", "control", "adjacent", "benign", "spleen")
COMMON_GENE_ALIASES = {
    "PD1": "PDCD1",
    "PD-1": "PDCD1",
    "TIM3": "HAVCR2",
    "TIM-3": "HAVCR2",
    "XBP-1": "XBP1",
}


@dataclass
class PairedModalityResult:
    """Container that behaves like both a tuple and a small mapping."""

    rna: object
    atac: object

    def __iter__(self):
        yield self.rna
        yield self.atac

    def __len__(self) -> int:
        return 2

    def __getitem__(self, item):
        if item in (0, "rna"):
            return self.rna
        if item in (1, "atac"):
            return self.atac
        raise KeyError(item)

    def __contains__(self, item) -> bool:
        return item in {"rna", "atac", 0, 1}

    def keys(self) -> tuple[str, str]:
        return ("rna", "atac")

    def items(self):
        return (("rna", self.rna), ("atac", self.atac))


def _gene_candidates(requested: str) -> list[str]:
    candidates = [requested]
    alias = COMMON_GENE_ALIASES.get(requested.upper())
    if alias and alias not in candidates:
        candidates.append(alias)
    return candidates


def ensure_obs_column(adata, column: str, fill_value: str = "Unknown", as_category: bool = True) -> None:
    if column not in adata.obs.columns:
        adata.obs[column] = fill_value
    else:
        adata.obs[column] = adata.obs[column].astype("object").where(adata.obs[column].notna(), fill_value)
    if as_category:
        adata.obs[column] = adata.obs[column].astype("category")


def ensure_obs_columns(adata, columns: Iterable[str], fill_value: str = "Unknown", as_category: bool = True) -> None:
    for column in columns:
        ensure_obs_column(adata, column, fill_value=fill_value, as_category=as_category)


def infer_tumor_like_tissues(adata, tissue_col: str = "tissue") -> list[str]:
    if tissue_col not in adata.obs.columns:
        raise KeyError(f"'{tissue_col}' is not present in adata.obs.")
    labels = adata.obs[tissue_col].dropna().astype(str).map(str.strip)
    inferred: list[str] = []
    for label in labels.value_counts(dropna=False).index.tolist():
        lowered = label.lower()
        if any(token in lowered for token in NON_TUMOR_HINT_TOKENS):
            continue
        if any(token in lowered for token in TUMOR_HINT_TOKENS):
            inferred.append(label)
    return inferred


def tumor_like_subset(adata, tissue_col: str = "tissue", copy: bool = True):
    tissues = infer_tumor_like_tissues(adata, tissue_col=tissue_col)
    if not tissues:
        raise ValueError(f"No tumor-like labels inferred from '{tissue_col}'.")
    mask = adata.obs[tissue_col].astype(str).isin(tissues)
    subset = adata[mask].copy() if copy else adata[mask]
    print(f"Tumor-like tissues inferred from {tissue_col}: {', '.join(tissues)}")
    print(f"Tumor-like subset cells: {subset.n_obs}")
    return subset


def resolve_gene_names(
    adata_or_genes,
    genes: Iterable[str] | None = None,
    var_fields: Iterable[str] = ("gene_name", "gene", "symbol"),
) -> dict[str, str]:
    if genes is None:
        if hasattr(adata_or_genes, "var_names") and hasattr(adata_or_genes, "obs"):
            return adata_or_genes
        gene_iterable = adata_or_genes
        try:
            return {str(gene).strip(): str(gene).strip() for gene in gene_iterable if str(gene).strip()}
        except TypeError:
            requested = str(adata_or_genes).strip()
            return {requested: requested} if requested else {}
    adata = adata_or_genes
    lookup = {str(name).upper(): str(name) for name in adata.var_names}
    for field in var_fields:
        if field in adata.var.columns:
            for idx, value in adata.var[field].astype(str).items():
                lookup.setdefault(value.upper(), str(idx))
    if "gene_activity_var_names" in adata.uns:
        for value in adata.uns["gene_activity_var_names"]:
            lookup.setdefault(str(value).upper(), str(value))
    resolved: dict[str, str] = {}
    for gene in genes:
        requested = str(gene).strip()
        if not requested:
            continue
        for candidate in _gene_candidates(requested):
            matched = lookup.get(candidate.upper())
            if matched:
                resolved[requested] = matched
                break
    return resolved


def resolve_atac_signal_names(adata_atac, genes: Iterable[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if "gene_activity_var_names" in adata_atac.uns:
        for value in adata_atac.uns["gene_activity_var_names"]:
            lookup[str(value).upper()] = str(value)
    for requested, matched in resolve_gene_names(adata_atac, genes).items():
        lookup.setdefault(requested.upper(), matched)
    resolved: dict[str, str] = {}
    for gene in genes:
        requested = str(gene).strip()
        if not requested:
            continue
        for candidate in _gene_candidates(requested):
            matched = lookup.get(candidate.upper())
            if matched:
                resolved[requested] = matched
                break
    return resolved


def expression_frame(adata, genes: Iterable[str], obs_columns: Iterable[str] | None = None) -> pd.DataFrame:
    resolved = resolve_gene_names(adata, genes)
    actual_features = list(dict.fromkeys(resolved.values()))
    if not actual_features:
        raise ValueError("None of the requested genes were found in the RNA object.")
    matrix = adata[:, actual_features].X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    frame = pd.DataFrame(matrix, index=adata.obs_names, columns=actual_features)
    if obs_columns:
        for column in obs_columns:
            if column in adata.obs.columns:
                frame[column] = adata.obs[column].values
    return frame


def atac_signal_frame(
    adata_atac,
    genes: Iterable[str] | None = None,
    *,
    features: Iterable[str] | None = None,
    mode: str | None = None,
    layer_preference: tuple[str, ...] = ("gene_activity", "counts"),
    obs_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    requested_features = genes if genes is not None else features
    if requested_features is None:
        raise ValueError("Provide genes or features when extracting ATAC signals.")
    if mode:
        lowered = str(mode).strip().lower()
        if lowered == "gene_activity":
            layer_preference = ("gene_activity", "counts")
        elif lowered in {"counts", "accessibility", "peaks"}:
            layer_preference = ("counts",)
    resolved = resolve_atac_signal_names(adata_atac, requested_features)
    if not resolved:
        raise ValueError("None of the requested genes were found in the ATAC object.")
    actual_features = list(dict.fromkeys(resolved.values()))
    if "gene_activity" in layer_preference and "gene_activity" in adata_atac.obsm and "gene_activity_var_names" in adata_atac.uns:
        gene_activity_names = [str(item) for item in adata_atac.uns["gene_activity_var_names"]]
        gene_to_idx = {name: idx for idx, name in enumerate(gene_activity_names)}
        selected_idx = [gene_to_idx[name] for name in actual_features if name in gene_to_idx]
        if not selected_idx:
            raise ValueError("Requested genes were not found in ATAC gene_activity.")
        matrix = adata_atac.obsm["gene_activity"][:, selected_idx]
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        frame = pd.DataFrame(matrix, index=adata_atac.obs_names, columns=[gene_activity_names[idx] for idx in selected_idx])
    else:
        layer_name = next((name for name in layer_preference if name in adata_atac.layers), None)
        matrix = adata_atac[:, actual_features].layers[layer_name] if layer_name else adata_atac[:, actual_features].X
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        frame = pd.DataFrame(matrix, index=adata_atac.obs_names, columns=actual_features)
    if obs_columns:
        for column in obs_columns:
            if column in adata_atac.obs.columns:
                frame[column] = adata_atac.obs[column].values
    return frame


def find_shared_obs_names(adata_rna, adata_atac) -> list[str]:
    return sorted(set(map(str, adata_rna.obs_names)).intersection(map(str, adata_atac.obs_names)))


def paired_modality_subset(
    adata_rna,
    adata_atac,
    obs_column: str | None = None,
    subset_categories: Iterable[str] | None = None,
    copy: bool = True,
):
    shared = find_shared_obs_names(adata_rna, adata_atac)
    if not shared:
        raise ValueError("No shared observation names were found between RNA and ATAC objects.")
    rna_subset = adata_rna[shared]
    atac_subset = adata_atac[shared]
    if obs_column and subset_categories is not None:
        categories = {str(item) for item in subset_categories}
        if obs_column in rna_subset.obs.columns:
            rna_mask = rna_subset.obs[obs_column].astype(str).isin(categories)
            rna_subset = rna_subset[rna_mask]
        if obs_column in atac_subset.obs.columns:
            atac_mask = atac_subset.obs[obs_column].astype(str).isin(categories)
            atac_subset = atac_subset[atac_mask]
        shared = find_shared_obs_names(rna_subset, atac_subset)
        if not shared:
            raise ValueError(f"No shared paired observations remained after filtering on '{obs_column}'.")
        rna_subset = rna_subset[shared]
        atac_subset = atac_subset[shared]
    if copy:
        rna_subset = rna_subset.copy()
        atac_subset = atac_subset.copy()
    return PairedModalityResult(rna=rna_subset, atac=atac_subset)


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
    ensure_obs_column(adata, groupby, fill_value="Unknown", as_category=True)
    counts = adata.obs[groupby].value_counts(dropna=False)
    eligible = counts[counts >= min_cells_per_group].index.astype(str).tolist()
    if len(eligible) < 2:
        raise ValueError(f"Not enough groups with >= {min_cells_per_group} observations for '{groupby}'.")
    working = adata[adata.obs[groupby].astype(str).isin(eligible)].copy()
    working.obs[groupby] = working.obs[groupby].astype(str).astype("category")
    selected_groups = None if groups is None else [item for item in groups if item in set(eligible)]
    if groups is not None and not selected_groups:
        raise ValueError(f"Requested groups are not eligible for '{groupby}'.")
    kwargs = {"groupby": groupby, "method": method}
    if selected_groups is not None:
        kwargs["groups"] = selected_groups
    if reference is not None:
        kwargs["reference"] = reference
    if layer is not None:
        kwargs["layer"] = layer
    if use_raw is not None:
        kwargs["use_raw"] = use_raw
    if key_added is not None:
        kwargs["key_added"] = key_added
    sc.tl.rank_genes_groups(working, **kwargs)
    return working


def safe_rank_features_groups(
    adata_atac,
    *,
    groupby: str,
    layer: str | None = None,
    min_cells_per_group: int = 20,
    key_added: str | None = None,
):
    return safe_rank_genes_groups(
        adata_atac,
        groupby=groupby,
        layer=layer,
        use_raw=False,
        min_cells_per_group=min_cells_per_group,
        key_added=key_added,
    )


def aggregate_group_signal(adata, *, groupby: str, genes: Iterable[str], signal_getter) -> pd.DataFrame:
    frame = signal_getter(adata, genes, obs_columns=[groupby])
    grouped = frame.groupby(groupby, observed=False)[[col for col in frame.columns if col != groupby]].mean()
    return grouped


def _signal_columns_from_frame(frame: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    excluded = {str(item) for item in exclude}
    return [
        str(col)
        for col in frame.columns
        if str(col) not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]


def _summarize_signal_frames(
    rna_frame: pd.DataFrame,
    atac_frame: pd.DataFrame,
    *,
    rna_groupby: str = "cell_type",
    atac_groupby: str = "cell_type",
) -> pd.DataFrame:
    if rna_groupby in rna_frame.columns and atac_groupby in atac_frame.columns:
        rna_grouped = rna_frame.groupby(rna_groupby, observed=False)[_signal_columns_from_frame(rna_frame, exclude=[rna_groupby])].mean()
        atac_grouped = atac_frame.groupby(atac_groupby, observed=False)[_signal_columns_from_frame(atac_frame, exclude=[atac_groupby])].mean()
        common_groups = rna_grouped.index.intersection(atac_grouped.index)
        common_genes = rna_grouped.columns.intersection(atac_grouped.columns)
        if common_groups.empty or common_genes.empty:
            return pd.DataFrame()
        rows: list[dict[str, object]] = []
        for group in common_groups:
            for gene in common_genes:
                rows.append(
                    {
                        "group": str(group),
                        "gene": str(gene),
                        "rna_mean": float(rna_grouped.loc[group, gene]),
                        "atac_mean": float(atac_grouped.loc[group, gene]),
                    }
                )
        return pd.DataFrame(rows)
    common_index = rna_frame.index.intersection(atac_frame.index)
    common_genes = pd.Index(_signal_columns_from_frame(rna_frame)).intersection(_signal_columns_from_frame(atac_frame))
    if common_index.empty or common_genes.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for gene in common_genes:
        rows.append(
            {
                "gene": str(gene),
                "n_shared_obs": int(len(common_index)),
                "rna_mean": float(rna_frame.loc[common_index, gene].mean()),
                "atac_mean": float(atac_frame.loc[common_index, gene].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_rna_atac_link(
    adata_rna,
    adata_atac,
    *,
    rna_groupby: str = "cell_type",
    atac_groupby: str = "cell_type",
    genes: Iterable[str] | None = None,
) -> pd.DataFrame:
    if isinstance(adata_rna, pd.DataFrame) and isinstance(adata_atac, pd.DataFrame):
        return _summarize_signal_frames(
            adata_rna,
            adata_atac,
            rna_groupby=rna_groupby,
            atac_groupby=atac_groupby,
        )
    if genes is None:
        raise ValueError("Provide genes when summarizing RNA-ATAC links from AnnData objects.")
    rna_grouped = aggregate_group_signal(adata_rna, groupby=rna_groupby, genes=genes, signal_getter=expression_frame)
    atac_grouped = aggregate_group_signal(adata_atac, groupby=atac_groupby, genes=genes, signal_getter=atac_signal_frame)
    common_groups = rna_grouped.index.intersection(atac_grouped.index)
    common_genes = rna_grouped.columns.intersection(atac_grouped.columns)
    if common_groups.empty or common_genes.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for group in common_groups:
        for gene in common_genes:
            rows.append(
                {
                    "group": str(group),
                    "gene": str(gene),
                    "rna_mean": float(rna_grouped.loc[group, gene]),
                    "atac_mean": float(atac_grouped.loc[group, gene]),
                }
            )
    return pd.DataFrame(rows)
