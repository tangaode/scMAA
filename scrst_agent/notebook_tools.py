"""Reusable notebook helper functions for scRNA + spatial analysis."""

from __future__ import annotations

import importlib.util
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

SPATIAL_MAPPING_METHODS = ("auto", "marker_transfer", "cell2location")


def cell2location_available() -> bool:
    return importlib.util.find_spec("cell2location") is not None and importlib.util.find_spec("scvi") is not None


def get_spatial_mapping_method(requested: str = "auto") -> str:
    normalized = str(requested or "auto").strip().lower().replace("-", "_")
    if normalized not in SPATIAL_MAPPING_METHODS:
        normalized = "auto"
    if normalized == "auto":
        return "marker_transfer"
    if normalized == "cell2location" and not cell2location_available():
        print("cell2location is not available. Falling back to marker_transfer.")
        return "marker_transfer"
    return normalized


def _clean_cell2location_label(label: object) -> str:
    text = str(label)
    prefixes = (
        "q05cell_abundance_w_sf_",
        "q05_cell_abundance_w_sf_",
        "means_cell_abundance_w_sf_",
        "means_per_cluster_mu_fg_",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):]
                changed = True
    return text


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
    print(f"Tumor-like subset cells/spots: {subset.n_obs}")
    return subset


def resolve_gene_names(adata, genes: Iterable[str]) -> dict[str, str]:
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
        matched = None
        for candidate in candidates:
            matched = var_lookup.get(candidate.upper())
            if matched:
                break
        if matched:
            resolved[requested] = matched
    return resolved


def expression_frame(adata, genes: Iterable[str], obs_columns: Iterable[str] | None = None) -> pd.DataFrame:
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
    return frame


def spatial_coordinate_frame(adata, coord_key: str = "spatial") -> pd.DataFrame:
    if coord_key not in adata.obsm:
        raise KeyError(f"'{coord_key}' is not present in adata.obsm.")
    coords = adata.obsm[coord_key]
    frame = pd.DataFrame(coords, index=adata.obs_names, columns=["spatial_x", "spatial_y"][: coords.shape[1]])
    return frame


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
    selected_groups = groups
    if selected_groups is not None:
        selected_groups = [item for item in groups if item in set(eligible)]
        if not selected_groups:
            raise ValueError(f"Requested groups are not eligible for '{groupby}'.")
    kwargs = {
        "groupby": groupby,
        "method": method,
    }
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


def transfer_reference_markers_to_spatial(
    adata_rna,
    adata_spatial,
    *args,
    cell_type_col: str = "cell_type",
    groupby: str | None = None,
    markers: dict | list | tuple | None = None,
    top_n: int = 20,
    prefix: str = "sig_",
    **_ignored_kwargs,
) -> pd.DataFrame:
    if args:
        first_arg = args[0]
        if isinstance(first_arg, str) and first_arg:
            cell_type_col = first_arg
    if groupby:
        cell_type_col = groupby
    ensure_obs_column(adata_rna, cell_type_col, fill_value="Unknown", as_category=True)
    shared_genes = set(map(str, adata_spatial.var_names))
    score_records: list[dict[str, object]] = []
    if markers is not None:
        if isinstance(markers, dict):
            marker_iter = markers.items()
        else:
            marker_iter = [(cell_type_col, markers)]
        for group, genes in marker_iter:
            gene_list = [str(gene) for gene in list(genes)[:top_n] if str(gene) in shared_genes]
            if not gene_list:
                continue
            score_name = f"{prefix}{group}"
            sc.tl.score_genes(adata_spatial, gene_list=gene_list, score_name=score_name, use_raw=True)
            score_records.append({"cell_type": str(group), "score_name": score_name, "n_genes": len(gene_list)})
        return pd.DataFrame(score_records)

    ranked = safe_rank_genes_groups(
        adata_rna,
        groupby=cell_type_col,
        min_cells_per_group=20,
        method="wilcoxon",
        key_added=f"{cell_type_col}_markers",
    )
    for group in ranked.obs[cell_type_col].cat.categories:
        marker_df = sc.get.rank_genes_groups_df(ranked, group=str(group), key=f"{cell_type_col}_markers").head(top_n)
        gene_list = [gene for gene in marker_df["names"].astype(str).tolist() if gene in shared_genes]
        if not gene_list:
            continue
        score_name = f"{prefix}{group}"
        sc.tl.score_genes(adata_spatial, gene_list=gene_list, score_name=score_name, use_raw=True)
        score_records.append({"cell_type": str(group), "score_name": score_name, "n_genes": len(gene_list)})
    return pd.DataFrame(score_records)


def run_cell2location_mapping(
    adata_rna,
    adata_spatial,
    *args,
    cell_type_col: str = "cell_type",
    groupby: str | None = None,
    max_epochs_reference: int = 120,
    max_epochs_mapping: int = 200,
    batch_key: str | None = None,
    prefix: str = "c2l_",
    **_ignored_kwargs,
) -> pd.DataFrame:
    if args:
        first_arg = args[0]
        if isinstance(first_arg, str) and first_arg:
            cell_type_col = first_arg
    if groupby:
        cell_type_col = groupby
    if not cell2location_available():
        raise ImportError("cell2location and scvi-tools are required for this mapping mode.")

    ensure_obs_column(adata_rna, cell_type_col, fill_value="Unknown", as_category=True)
    if "counts" not in adata_rna.layers:
        raise ValueError("RNA object requires a 'counts' layer for cell2location.")
    if "counts" not in adata_spatial.layers:
        raise ValueError("Spatial object requires a 'counts' layer for cell2location.")

    from cell2location.models import Cell2location, RegressionModel

    rna = adata_rna.copy()
    spatial = adata_spatial.copy()
    rna.var_names_make_unique()
    spatial.var_names_make_unique()

    RegressionModel.setup_anndata(rna, layer="counts", labels_key=cell_type_col)
    reg_model = RegressionModel(rna)
    reg_model.train(max_epochs=max_epochs_reference, train_size=1)
    rna = reg_model.export_posterior(rna, sample_kwargs={"num_samples": 200, "batch_size": 2500})
    if "means_per_cluster_mu_fg" in rna.varm:
        signatures = rna.varm["means_per_cluster_mu_fg"]
    elif "means_per_cluster_mu_fg" in rna.var:
        signatures = rna.var["means_per_cluster_mu_fg"]
    else:
        raise ValueError("cell2location reference signatures were not exported from the RNA model.")
    if not isinstance(signatures, pd.DataFrame):
        signatures = pd.DataFrame(signatures, index=rna.var_names)

    common = signatures.index.intersection(spatial.var_names)
    signatures = signatures.loc[common]
    spatial = spatial[:, common].copy()

    effective_batch_key = batch_key if batch_key and batch_key in spatial.obs.columns else None
    if effective_batch_key is not None:
        Cell2location.setup_anndata(spatial, layer="counts", batch_key=effective_batch_key)
    else:
        Cell2location.setup_anndata(spatial, layer="counts")

    mapping_model = Cell2location(
        spatial,
        cell_state_df=signatures,
        N_cells_per_location=10,
        detection_alpha=20,
    )
    mapping_model.train(max_epochs=max_epochs_mapping, train_size=1)
    spatial = mapping_model.export_posterior(
        spatial,
        sample_kwargs={"num_samples": 200, "batch_size": 2500},
    )

    abundance_df = None
    for key in ("q05_cell_abundance_w_sf", "means_cell_abundance_w_sf"):
        if key in spatial.obsm:
            abundance_df = spatial.obsm[key].copy()
            break
    if abundance_df is None:
        raise ValueError("cell2location did not export cell abundance estimates into adata_spatial.obsm.")

    if not isinstance(abundance_df, pd.DataFrame):
        abundance_df = pd.DataFrame(abundance_df, index=spatial.obs_names)
    abundance_df = abundance_df.loc[adata_spatial.obs_names]
    score_records: list[dict[str, object]] = []
    for column in abundance_df.columns:
        cleaned_label = _clean_cell2location_label(column)
        score_name = f"{prefix}{cleaned_label}"
        adata_spatial.obs[score_name] = abundance_df[column].astype(float).values
        score_records.append({"cell_type": cleaned_label, "score_name": score_name, "n_genes": np.nan})
    adata_spatial.uns["spatial_mapping_method"] = "cell2location"
    return pd.DataFrame(score_records)


def run_reference_mapping(
    adata_rna,
    adata_spatial,
    *args,
    method: str = "auto",
    **kwargs,
) -> pd.DataFrame:
    effective_method = get_spatial_mapping_method(method)
    if effective_method == "cell2location":
        return run_cell2location_mapping(adata_rna, adata_spatial, *args, **kwargs)
    return transfer_reference_markers_to_spatial(adata_rna, adata_spatial, *args, **kwargs)


def spatial_domain_de(
    adata_spatial,
    *,
    domain_col: str = "leiden",
    top_n: int = 10,
    min_spots_per_group: int = 20,
) -> pd.DataFrame:
    ranked = safe_rank_genes_groups(
        adata_spatial,
        groupby=domain_col,
        min_cells_per_group=min_spots_per_group,
        method="wilcoxon",
        key_added=f"{domain_col}_markers",
    )
    frames: list[pd.DataFrame] = []
    for group in ranked.obs[domain_col].cat.categories:
        frame = sc.get.rank_genes_groups_df(ranked, group=str(group), key=f"{domain_col}_markers").head(top_n)
        frame.insert(0, "domain", str(group))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
