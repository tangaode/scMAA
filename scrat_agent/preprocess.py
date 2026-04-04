"""Raw-data preparation pipeline for scRNA + scATAC analysis."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tarfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.io import mmread
from sklearn.decomposition import TruncatedSVD

from scrt_agent.preprocess import (
    _annotate_clusters_with_llm,
    _discover_from_directory,
    _extract_marker_table,
    _read_table,
    _sample_qc_summary,
    _save_umap_figure,
)

from .logger import AgentLogger
from .utils import read_text

try:
    import gzip
except Exception:  # pragma: no cover
    gzip = None

try:
    import litellm

    litellm.drop_params = True
except Exception:  # pragma: no cover
    litellm = None

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


@dataclass
class MultiomeInput:
    sample_key: str
    sample_id: str
    tissue: str
    source_type: str
    source_path: Path | None = None
    barcodes_path: Path | None = None
    features_path: Path | None = None
    matrix_path: Path | None = None


@dataclass
class PreparationResult:
    output_dir: Path
    rna_h5ad_path: Path
    atac_h5ad_path: Path
    rna_markers_path: Path
    atac_markers_path: Path
    rna_annotations_path: Path
    atac_annotations_path: Path
    qc_summary_path: Path
    manifest_path: Path
    rna_umap_path: Path
    atac_umap_path: Path


def _load_environment_files(*paths: Path) -> None:
    if load_dotenv is None:
        return
    seen: set[Path] = set()
    for path in paths:
        if not path:
            continue
        directory = path.resolve()
        if directory.is_file():
            directory = directory.parent
        if directory in seen:
            continue
        seen.add(directory)
        for name in (".env", "OPENAI.env", "deepseek.env"):
            env_path = directory / name
            if env_path.exists():
                load_dotenv(env_path, override=False)


def _parse_sample_key(sample_key: str) -> tuple[str, str]:
    tokens = sample_key.replace("-", "_").split("_")
    if len(tokens) <= 1:
        return sample_key, "unknown"
    return tokens[0], "_".join(tokens[1:])


def _stage_input(raw_input_path: str | Path, stage_dir: Path, logger: AgentLogger) -> Path:
    raw_path = Path(raw_input_path).resolve()
    if raw_path.is_dir():
        logger.info(f"Using raw input directory: {raw_path}")
        return raw_path
    if tarfile.is_tarfile(raw_path):
        extract_dir = stage_dir / f"{raw_path.stem}_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        if any(extract_dir.iterdir()):
            logger.info(f"Using existing extracted raw files in {extract_dir}")
            return extract_dir
        logger.info(f"Extracting raw archive {raw_path} to {extract_dir}")
        with tarfile.open(raw_path, "r") as handle:
            handle.extractall(extract_dir)
        return extract_dir
    raise ValueError(f"Unsupported raw input path: {raw_input_path}")


def _infer_sample_key_from_source(path: Path) -> str:
    stem = path.stem
    for suffix in (
        "_gex_raw_feature_bc_matrix",
        "_raw_feature_bc_matrix",
        "_filtered_feature_bc_matrix",
        "_feature_bc_matrix",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if stem.startswith("GSM") and "_" in stem:
        parts = stem.split("_", 1)
        if len(parts) == 2 and parts[1]:
            return parts[1]
    if path.name in {"filtered_feature_bc_matrix", "filtered_feature_bc_matrix.h5"}:
        parent = path.parent
        if parent.name == "outs" and parent.parent.name:
            return parent.parent.name
        if parent.name:
            return parent.name
    if path.name == "outs" and path.parent.name:
        return path.parent.name
    if path.stem == "filtered_feature_bc_matrix":
        return path.parent.name or path.stem
    return stem if path.is_file() else path.name


def _discover_standard_multiome_inputs(root: Path) -> list[MultiomeInput]:
    results: list[MultiomeInput] = []
    seen: set[Path] = set()

    def add_candidate(source_type: str, path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        sample_key = _infer_sample_key_from_source(path)
        sample_id, tissue = _parse_sample_key(sample_key)
        results.append(
            MultiomeInput(
                sample_key=sample_key,
                sample_id=sample_id,
                tissue=tissue,
                source_type=source_type,
                source_path=resolved,
            )
        )

    for path in [root / "filtered_feature_bc_matrix.h5", root / "outs" / "filtered_feature_bc_matrix.h5"]:
        if path.exists():
            add_candidate("h5", path)
    for path in [root / "filtered_feature_bc_matrix", root / "outs" / "filtered_feature_bc_matrix"]:
        if path.is_dir() and (path / "matrix.mtx.gz").exists():
            add_candidate("dir", path)
    for path in root.rglob("filtered_feature_bc_matrix.h5"):
        add_candidate("h5", path)
    for path in root.rglob("filtered_feature_bc_matrix"):
        if path.is_dir() and (path / "matrix.mtx.gz").exists():
            add_candidate("dir", path)
    for pattern in ("*_gex_raw_feature_bc_matrix.h5", "*_raw_feature_bc_matrix.h5", "*_filtered_feature_bc_matrix.h5", "*_feature_bc_matrix.h5"):
        for path in root.rglob(pattern):
            add_candidate("h5", path)
    return results


def _discover_triplet_multiome_inputs(root: Path) -> list[MultiomeInput]:
    results: list[MultiomeInput] = []
    for sample in _discover_from_directory(root).values():
        if not sample.barcodes_path or not sample.features_path or not sample.matrix_path:
            continue
        results.append(
            MultiomeInput(
                sample_key=sample.sample_key,
                sample_id=sample.sample_id,
                tissue=sample.tissue,
                source_type="triplet",
                barcodes_path=sample.barcodes_path,
                features_path=sample.features_path,
                matrix_path=sample.matrix_path,
            )
        )
    return results


def _feature_type_masks(feature_type: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    values = feature_type.fillna("").astype(str)
    gene_mask = values.str.casefold().eq("gene expression")
    peak_mask = values.str.contains("peak", case=False) | values.str.contains("chromatin", case=False)
    return gene_mask.to_numpy(), peak_mask.to_numpy()


def _annotate_peak_columns(var: pd.DataFrame) -> None:
    feature_names = var["feature_name"].astype(str) if "feature_name" in var.columns else pd.Series(var.index.astype(str), index=var.index)
    parsed = feature_names.str.extract(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
    if not parsed.empty:
        if "chrom" not in var.columns:
            var["chrom"] = parsed["chrom"].fillna("").astype(str).to_numpy()
        if "start" not in var.columns:
            var["start"] = pd.to_numeric(parsed["start"], errors="coerce")
        if "end" not in var.columns:
            var["end"] = pd.to_numeric(parsed["end"], errors="coerce")


def _make_obs_names(sample: MultiomeInput, barcodes: list[str]) -> pd.Index:
    return pd.Index([f"{sample.sample_key}:{barcode}" for barcode in barcodes])


def _ensure_csr(matrix) -> sp.csr_matrix:
    if sp.issparse(matrix):
        return matrix.tocsr()
    return sp.csr_matrix(matrix)


def _n_nonzero_by_row(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray(matrix.getnnz(axis=1)).ravel()
    return np.asarray((matrix > 0).sum(axis=1)).ravel()


def _split_matrix_to_modalities(
    *,
    matrix: sp.csr_matrix,
    features: pd.DataFrame,
    barcodes: list[str],
    sample: MultiomeInput,
) -> tuple[ad.AnnData, ad.AnnData]:
    feature_columns = ["feature_id", "feature_name", "feature_type"][: features.shape[1]]
    features = features.copy()
    features.columns = feature_columns
    if "feature_name" not in features.columns:
        features["feature_name"] = features.iloc[:, 0].astype(str)
    if "feature_type" not in features.columns:
        raise ValueError("The feature table does not include feature_type, so RNA and ATAC modalities cannot be split.")

    gene_mask, peak_mask = _feature_type_masks(features["feature_type"])
    if gene_mask.sum() == 0 or peak_mask.sum() == 0:
        raise ValueError("Could not identify both Gene Expression and Peak features from the input.")

    obs_names = _make_obs_names(sample, barcodes)
    obs = pd.DataFrame(
        {
            "barcode": barcodes,
            "sample_key": sample.sample_key,
            "sample_id": sample.sample_id,
            "tissue": sample.tissue,
        },
        index=obs_names,
    )

    rna_features = features.loc[gene_mask].reset_index(drop=True)
    rna = ad.AnnData(X=matrix[gene_mask, :].T.tocsr(), obs=obs.copy())
    rna.var["gene_id"] = rna_features["feature_id"].astype(str).to_numpy() if "feature_id" in rna_features.columns else rna_features["feature_name"].astype(str).to_numpy()
    rna.var["gene_name"] = rna_features["feature_name"].astype(str).to_numpy()
    rna.var["feature_type"] = rna_features["feature_type"].astype(str).to_numpy()
    rna.var_names = pd.Index(rna.var["gene_name"].astype(str))
    rna.var_names_make_unique()
    rna.var_names.name = None

    peak_features = features.loc[peak_mask].reset_index(drop=True)
    atac = ad.AnnData(X=matrix[peak_mask, :].T.tocsr(), obs=obs.copy())
    atac.var["feature_id"] = peak_features["feature_id"].astype(str).to_numpy() if "feature_id" in peak_features.columns else peak_features["feature_name"].astype(str).to_numpy()
    atac.var["feature_name"] = peak_features["feature_name"].astype(str).to_numpy()
    atac.var["feature_type"] = peak_features["feature_type"].astype(str).to_numpy()
    atac.var_names = pd.Index(atac.var["feature_name"].astype(str))
    atac.var_names_make_unique()
    atac.var_names.name = None
    _annotate_peak_columns(atac.var)
    return rna, atac


def _read_multiome_triplet(sample: MultiomeInput) -> tuple[ad.AnnData, ad.AnnData]:
    if gzip is None:
        raise RuntimeError("gzip module is unavailable.")
    if not sample.barcodes_path or not sample.features_path or not sample.matrix_path:
        raise ValueError(f"Incomplete triplet files for sample {sample.sample_key}")
    barcodes = _read_table(sample.barcodes_path, sep="\t", header=None).iloc[:, 0].astype(str).tolist()
    features = _read_table(sample.features_path, sep="\t", header=None)
    with gzip.open(sample.matrix_path, "rb") as handle:
        matrix = mmread(handle).tocsr()
    if matrix.shape[1] != len(barcodes):
        raise ValueError(f"Barcode count does not match matrix columns for sample {sample.sample_key}")
    return _split_matrix_to_modalities(matrix=matrix, features=features, barcodes=barcodes, sample=sample)


def _read_multiome_directory(sample: MultiomeInput) -> tuple[ad.AnnData, ad.AnnData]:
    source = Path(sample.source_path or "")
    features = _read_table(source / "features.tsv.gz", sep="\t", header=None)
    barcodes = _read_table(source / "barcodes.tsv.gz", sep="\t", header=None).iloc[:, 0].astype(str).tolist()
    if gzip is None:
        raise RuntimeError("gzip module is unavailable.")
    with gzip.open(source / "matrix.mtx.gz", "rb") as handle:
        matrix = mmread(handle).tocsr()
    return _split_matrix_to_modalities(matrix=matrix, features=features, barcodes=barcodes, sample=sample)


def _read_multiome_h5(sample: MultiomeInput) -> tuple[ad.AnnData, ad.AnnData]:
    source = Path(sample.source_path or "")
    try:
        full = sc.read_10x_h5(source, gex_only=False)
    except TypeError:
        full = sc.read_10x_h5(source)
    var = full.var.copy()
    feature_type_col = next((col for col in var.columns if str(col).lower() in {"feature_types", "feature_type"}), None)
    if feature_type_col is None:
        raise ValueError("The 10x H5 file does not include feature_types, so RNA and ATAC modalities cannot be split.")
    var["feature_name"] = full.var_names.astype(str)
    gene_mask, peak_mask = _feature_type_masks(var[feature_type_col])
    if gene_mask.sum() == 0 or peak_mask.sum() == 0:
        raise ValueError("Could not identify both Gene Expression and Peak features from the H5 file.")

    barcodes = full.obs_names.astype(str).tolist()
    full.obs_names = _make_obs_names(sample, barcodes)
    full.obs["barcode"] = barcodes
    full.obs["sample_key"] = sample.sample_key
    full.obs["sample_id"] = sample.sample_id
    full.obs["tissue"] = sample.tissue

    rna = full[:, gene_mask].copy()
    if "gene_name" not in rna.var.columns:
        rna.var["gene_name"] = rna.var_names.astype(str)
    if "gene_id" not in rna.var.columns:
        gene_id_col = next((col for col in rna.var.columns if str(col).lower() in {"gene_ids", "gene_id", "id"}), None)
        rna.var["gene_id"] = rna.var[gene_id_col].astype(str).to_numpy() if gene_id_col else rna.var_names.astype(str)
    rna.var["feature_type"] = var.loc[gene_mask, feature_type_col].astype(str).to_numpy()
    rna.var_names = pd.Index(rna.var["gene_name"].astype(str))
    rna.var_names_make_unique()
    rna.var_names.name = None

    atac = full[:, peak_mask].copy()
    atac.var["feature_name"] = atac.var_names.astype(str)
    atac.var["feature_type"] = var.loc[peak_mask, feature_type_col].astype(str).to_numpy()
    if "feature_id" not in atac.var.columns:
        feat_id_col = next((col for col in atac.var.columns if str(col).lower() in {"gene_ids", "feature_id", "id"}), None)
    atac.var["feature_id"] = atac.var[feat_id_col].astype(str).to_numpy() if feat_id_col else atac.var_names.astype(str)
    atac.var_names = pd.Index(atac.var["feature_name"].astype(str))
    atac.var_names_make_unique()
    atac.var_names.name = None
    _annotate_peak_columns(atac.var)
    return rna, atac


def _load_multiome_sample(sample: MultiomeInput) -> tuple[ad.AnnData, ad.AnnData]:
    if sample.source_type == "triplet":
        return _read_multiome_triplet(sample)
    if sample.source_type == "dir":
        return _read_multiome_directory(sample)
    if sample.source_type == "h5":
        return _read_multiome_h5(sample)
    raise ValueError(f"Unsupported source type: {sample.source_type}")


def _run_atac_lsi(adata: ad.AnnData, *, n_components: int = 30, n_neighbors: int = 15, leiden_resolution: float = 0.8) -> None:
    matrix = _ensure_csr(adata.layers["counts"] if "counts" in adata.layers else adata.X)
    cell_sums = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    feature_sums = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
    cell_sums[cell_sums == 0] = 1.0
    feature_sums[feature_sums == 0] = 1.0

    tf = sp.diags(1.0 / cell_sums) @ matrix
    idf = np.log1p(matrix.shape[0] / feature_sums)
    tfidf = tf @ sp.diags(idf)

    max_components = max(2, min(n_components, matrix.shape[0] - 1, matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=max_components, random_state=0)
    lsi = svd.fit_transform(tfidf)
    adata.obsm["X_lsi"] = lsi[:, 1:] if lsi.shape[1] > 2 else lsi
    sc.pp.neighbors(adata, use_rep="X_lsi", n_neighbors=min(n_neighbors, max(2, adata.n_obs - 1)))
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=leiden_resolution, key_added="leiden")


def _make_atac_annotation_table(adata_atac: ad.AnnData, adata_rna: ad.AnnData) -> pd.DataFrame:
    shared = sorted(set(map(str, adata_atac.obs_names)).intersection(map(str, adata_rna.obs_names)))
    groups = [str(item) for item in adata_atac.obs["leiden"].astype(str).unique().tolist()]
    if not shared:
        return pd.DataFrame(
            [
                {
                    "cluster_id": cluster,
                    "cell_type": f"ATAC cluster {cluster}",
                    "confidence": "low",
                    "rationale": "No shared barcodes were available to transfer labels from RNA clusters.",
                    "supporting_markers": "",
                }
                for cluster in sorted(groups, key=lambda x: (len(x), x))
            ]
        )

    frame = pd.DataFrame(
        {
            "cluster_id": adata_atac.obs.loc[shared, "leiden"].astype(str).values,
            "rna_label": adata_rna.obs.loc[shared, "cluster_cell_type"].astype(str).values,
        }
    )
    records = []
    for cluster, subset in frame.groupby("cluster_id", observed=True):
        counts = subset["rna_label"].value_counts()
        label = counts.index[0]
        fraction = float(counts.iloc[0] / counts.sum())
        confidence = "high" if fraction >= 0.7 else "medium" if fraction >= 0.5 else "low"
        records.append(
            {
                "cluster_id": str(cluster),
                "cell_type": str(label),
                "confidence": confidence,
                "rationale": f"Transferred by majority RNA label among paired cells in ATAC cluster {cluster} ({fraction:.2f} agreement).",
                "supporting_markers": "",
            }
        )
    return pd.DataFrame(records)


def _parse_interval_strings(values: pd.Series) -> pd.DataFrame:
    parsed = values.astype(str).str.extract(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
    parsed["start"] = pd.to_numeric(parsed["start"], errors="coerce")
    parsed["end"] = pd.to_numeric(parsed["end"], errors="coerce")
    return parsed


def _build_gene_activity_matrix(
    adata_rna: ad.AnnData,
    adata_atac: ad.AnnData,
    *,
    promoter_padding: int = 2000,
    bin_size: int = 50000,
) -> tuple[sp.csr_matrix, list[str]]:
    if "interval" not in adata_rna.var.columns:
        raise ValueError("RNA var does not contain genomic intervals required for gene_activity construction.")

    gene_names = adata_rna.var["gene_name"].astype(str) if "gene_name" in adata_rna.var.columns else pd.Series(adata_rna.var_names.astype(str), index=adata_rna.var_names)
    gene_df = pd.DataFrame(
        {
            "gene_name": gene_names.astype(str).values,
            "interval": adata_rna.var["interval"].astype(str).values,
        }
    )
    gene_coords = _parse_interval_strings(gene_df["interval"])
    gene_df = pd.concat([gene_df, gene_coords], axis=1).dropna(subset=["chrom", "start", "end"])
    gene_df["start"] = gene_df["start"].astype(int).clip(lower=0)
    gene_df["end"] = gene_df["end"].astype(int)
    gene_df["start"] = (gene_df["start"] - promoter_padding).clip(lower=0)
    gene_df["end"] = gene_df["end"] + promoter_padding
    gene_df = gene_df.drop_duplicates(subset=["gene_name"], keep="first").reset_index(drop=True)
    if gene_df.empty:
        raise ValueError("No valid RNA gene intervals were available for gene_activity construction.")

    peak_series = adata_atac.var["feature_name"].astype(str) if "feature_name" in adata_atac.var.columns else pd.Series(adata_atac.var_names.astype(str), index=adata_atac.var_names)
    peak_df = pd.DataFrame({"feature_name": peak_series.astype(str).values, "peak_idx": np.arange(adata_atac.n_vars)})
    peak_coords = _parse_interval_strings(peak_df["feature_name"])
    peak_df = pd.concat([peak_df, peak_coords], axis=1).dropna(subset=["chrom", "start", "end"])
    peak_df["start"] = peak_df["start"].astype(int).clip(lower=0)
    peak_df["end"] = peak_df["end"].astype(int)
    if peak_df.empty:
        raise ValueError("No valid ATAC peak intervals were available for gene_activity construction.")

    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    gene_names_order = gene_df["gene_name"].astype(str).tolist()

    for chrom, chr_genes in gene_df.groupby("chrom", observed=True):
        chr_peaks = peak_df.loc[peak_df["chrom"] == chrom].copy()
        if chr_peaks.empty:
            continue
        bin_map: dict[int, list[tuple[int, int, int]]] = {}
        for peak_idx, peak_start, peak_end in chr_peaks[["peak_idx", "start", "end"]].itertuples(index=False):
            start_bin = int(peak_start // bin_size)
            end_bin = int(peak_end // bin_size)
            for bin_id in range(start_bin, end_bin + 1):
                bin_map.setdefault(bin_id, []).append((int(peak_idx), int(peak_start), int(peak_end)))

        for gene_idx, gene_start, gene_end in chr_genes[["start", "end"]].itertuples(index=True):
            candidate_bins = range(int(gene_start // bin_size), int(gene_end // bin_size) + 1)
            candidates: dict[int, tuple[int, int]] = {}
            for bin_id in candidate_bins:
                for peak_idx, peak_start, peak_end in bin_map.get(bin_id, []):
                    candidates.setdefault(peak_idx, (peak_start, peak_end))
            if not candidates:
                continue
            overlaps = [peak_idx for peak_idx, (peak_start, peak_end) in candidates.items() if peak_start <= gene_end and peak_end >= gene_start]
            if not overlaps:
                continue
            rows.extend(overlaps)
            cols.extend([int(gene_idx)] * len(overlaps))
            data.extend([1] * len(overlaps))

    if not rows:
        raise ValueError("No peak-to-gene overlaps were found for gene_activity construction.")

    incidence = sp.csr_matrix((data, (rows, cols)), shape=(adata_atac.n_vars, len(gene_names_order)))
    atac_counts = _ensure_csr(adata_atac.layers["counts"] if "counts" in adata_atac.layers else adata_atac.X)
    gene_activity = (atac_counts @ incidence).tocsr()
    gene_activity.eliminate_zeros()
    return gene_activity, gene_names_order


def _list_input_files(root: Path, limit: int = 200) -> str:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            try:
                files.append(str(path.relative_to(root)))
            except Exception:
                files.append(str(path))
        if len(files) >= limit:
            break
    return "\n".join(files) if files else "No files found."


def _write_fallback_guidance(
    *,
    raw_input_path: Path,
    output_dir: Path,
    logger: AgentLogger,
    model_name: str,
    reason: str,
) -> tuple[Path, Path]:
    advice_path = output_dir / "unsupported_input_advice.txt"
    code_path = output_dir / "suggested_preprocessing.py"
    file_listing = _list_input_files(raw_input_path)
    generic_advice = (
        "The current scRNA-ATAC preprocessing pipeline supports common 10x multiome inputs such as\n"
        "`filtered_feature_bc_matrix.h5`, `filtered_feature_bc_matrix/`, or GEO-style flattened matrix triplets.\n\n"
        f"Detected issue:\n{reason}\n\n"
        "Please check whether the raw folder contains a standard Cell Ranger ARC output or a matrix that includes both\n"
        "`Gene Expression` and `Peaks` feature types. If the data is more custom, use the suggested code file as a starting point\n"
        "and then provide the processed RNA and ATAC h5ad files to the agent.\n\n"
        "Detected files:\n"
        f"{file_listing}\n"
    )
    code_template = (
        "import scanpy as sc\n"
        "import anndata as ad\n"
        "import pandas as pd\n\n"
        "# Replace this template with dataset-specific loading logic.\n"
        "# Goal:\n"
        "# 1. create processed_rna.h5ad with RNA cells x genes\n"
        "# 2. create processed_atac.h5ad with ATAC cells x peaks or gene-activity features\n"
        "# 3. keep shared barcodes if the data is paired\n"
        "# 4. include obs columns such as sample_key, sample_id, tissue, leiden, cluster_cell_type when possible\n"
    )

    generated = False
    if litellm is not None and os.environ.get("OPENAI_API_KEY"):
        prompt = (
            "You are helping preprocess raw single-cell RNA + ATAC data.\n"
            "The built-in pipeline could not parse the dataset.\n"
            "Write two sections.\n"
            "Section 1: short practical preprocessing advice.\n"
            "Section 2: a Python code skeleton that prepares processed_rna.h5ad and processed_atac.h5ad.\n\n"
            f"Failure reason:\n{reason}\n\n"
            f"Detected files under the raw input directory:\n{file_listing}\n"
        )
        try:
            response = litellm.completion(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You provide preprocessing advice for scRNA + scATAC data."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content or ""
            if text.strip():
                advice_path.write_text(text, encoding="utf-8")
                if "```python" in text:
                    code = text.split("```python", 1)[1].split("```", 1)[0].strip()
                    code_path.write_text(code + "\n", encoding="utf-8")
                else:
                    code_path.write_text(code_template, encoding="utf-8")
                generated = True
                logger.info(f"Wrote LLM fallback advice to {advice_path}")
        except Exception as exc:
            logger.warning(f"Could not generate LLM fallback preprocessing advice: {exc}")

    if not generated:
        advice_path.write_text(generic_advice, encoding="utf-8")
        code_path.write_text(code_template, encoding="utf-8")
    return advice_path, code_path


def _extract_python_block(text: str) -> str:
    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text.strip()


def _generate_fallback_script(
    *,
    raw_input_path: Path,
    work_dir: Path,
    model_name: str,
    logger: AgentLogger,
    reason: str,
    previous_script: str = "",
    previous_error: str = "",
) -> str:
    if litellm is None or not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("LLM fallback requires litellm and OPENAI_API_KEY.")
    file_listing = _list_input_files(raw_input_path)
    prompt = f"""
Write executable Python only.

Task:
- Inspect the raw input directory and load one scRNA object and one scATAC object from the files.
- Save them as:
  - {work_dir / 'fallback_rna_raw.h5ad'}
  - {work_dir / 'fallback_atac_raw.h5ad'}
- The outputs must be AnnData objects with cells in obs and genes or features in var.
- If the data is paired multiome, preserve matching barcodes across RNA and ATAC obs_names when possible.
- Add obs columns when possible: barcode, sample_key, sample_id, tissue.
- Keep raw counts in X or a counts layer.
- Use only local files. No network calls. No package installs. No explanations.

Failure reason:
{reason}

Detected files:
{file_listing}

Raw input directory:
{raw_input_path}

Working directory for outputs:
{work_dir}

If the previous attempt failed, fix it instead of rewriting from scratch.

Previous script:
{previous_script or 'None'}

Previous error:
{previous_error or 'None'}
""".strip()
    logger.log_prompt("user", prompt, "scrat_fallback_preprocess")
    response = litellm.completion(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write robust Python preprocessing scripts for scRNA + scATAC data. "
                    "Return Python only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    text = response.choices[0].message.content or ""
    script = _extract_python_block(text)
    logger.log_response(script, "scrat_fallback_preprocess_script")
    if not script.strip():
        raise RuntimeError("The fallback preprocessing model returned empty code.")
    return script + ("\n" if not script.endswith("\n") else "")


def _load_llm_fallback_outputs(work_dir: Path) -> tuple[ad.AnnData, ad.AnnData]:
    rna_path = work_dir / "fallback_rna_raw.h5ad"
    atac_path = work_dir / "fallback_atac_raw.h5ad"
    if not rna_path.exists() or not atac_path.exists():
        raise FileNotFoundError("Fallback preprocessing script did not create both fallback_rna_raw.h5ad and fallback_atac_raw.h5ad.")
    rna = sc.read_h5ad(rna_path)
    atac = sc.read_h5ad(atac_path)
    rna.obs_names = pd.Index(rna.obs_names.astype(str))
    atac.obs_names = pd.Index(atac.obs_names.astype(str))
    if "barcode" not in rna.obs.columns:
        rna.obs["barcode"] = [name.split(":")[-1] for name in rna.obs_names.astype(str)]
    if "barcode" not in atac.obs.columns:
        atac.obs["barcode"] = [name.split(":")[-1] for name in atac.obs_names.astype(str)]
    for adata in (rna, atac):
        for column in ("sample_key", "sample_id", "tissue"):
            if column not in adata.obs.columns:
                adata.obs[column] = "unknown"
        adata.var_names_make_unique()
    return rna, atac


def _try_llm_fallback_preprocessing(
    *,
    raw_input_path: Path,
    output_dir: Path,
    logger: AgentLogger,
    model_name: str,
    reason: str,
    max_attempts: int = 3,
) -> tuple[ad.AnnData, ad.AnnData] | None:
    if litellm is None or not os.environ.get("OPENAI_API_KEY"):
        return None
    work_dir = output_dir / "llm_fallback_preprocess"
    work_dir.mkdir(parents=True, exist_ok=True)
    previous_script = ""
    previous_error = ""
    for attempt in range(1, max_attempts + 1):
        script = _generate_fallback_script(
            raw_input_path=raw_input_path,
            work_dir=work_dir,
            model_name=model_name,
            logger=logger,
            reason=reason,
            previous_script=previous_script,
            previous_error=previous_error,
        )
        script_path = work_dir / f"attempt_{attempt}.py"
        script_path.write_text(script, encoding="utf-8")
        command = [sys.executable, str(script_path)]
        result = subprocess.run(
            command,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        logger.info(
            textwrap.dedent(
                f"""
                LLM fallback preprocessing attempt {attempt}
                Command: {command}
                Return code: {result.returncode}
                STDOUT:
                {result.stdout}

                STDERR:
                {result.stderr}
                """
            ).strip()
        )
        if result.returncode == 0:
            try:
                return _load_llm_fallback_outputs(work_dir)
            except Exception as exc:
                previous_error = f"Execution succeeded but output loading failed: {exc}"
                previous_script = script
                continue
        previous_error = result.stderr.strip() or result.stdout.strip() or f"Attempt {attempt} failed."
        previous_script = script
    return None


def prepare_dataset(
    *,
    raw_input_path: str,
    output_dir: str,
    annotation_model: str = "gpt-4o",
    annotation_notes_path: str | None = None,
    min_genes: int = 200,
    min_cells: int = 3,
    max_pct_mt: float = 15.0,
    min_atac_features: int = 1000,
    n_top_genes: int = 3000,
    n_pcs: int = 30,
    n_neighbors: int = 15,
    leiden_resolution: float = 0.8,
    marker_top_n: int = 100,
    annotation_marker_top_n: int = 50,
    log_prompts: bool = False,
) -> PreparationResult:
    outdir = Path(output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    _load_environment_files(Path(raw_input_path), outdir, Path.cwd())
    logger = AgentLogger("prepare_scrat_data", outdir / "logs", log_prompts=log_prompts)
    sc.settings.verbosity = 2
    sc.settings.set_figure_params(dpi=120, facecolor="white", frameon=False)

    staged_input = _stage_input(raw_input_path, outdir / "work", logger)
    triplet_inputs = _discover_triplet_multiome_inputs(staged_input)
    standard_inputs = _discover_standard_multiome_inputs(staged_input)
    seen_keys = {(item.sample_key, item.source_type, str(item.source_path or item.matrix_path)) for item in triplet_inputs}
    multiome_inputs = list(triplet_inputs)
    for item in standard_inputs:
        marker = (item.sample_key, item.source_type, str(item.source_path or item.matrix_path))
        if marker not in seen_keys:
            multiome_inputs.append(item)
    if not multiome_inputs:
        fallback_result = _try_llm_fallback_preprocessing(
            raw_input_path=staged_input,
            output_dir=outdir,
            logger=logger,
            model_name=annotation_model,
            reason="No supported 10x multiome input was found.",
        )
        if fallback_result is not None:
            rna_parts = [fallback_result[0]]
            atac_parts = [fallback_result[1]]
            sample_rows = [
                {
                    "sample_key": "llm_fallback",
                    "sample_id": "llm_fallback",
                    "tissue": "unknown",
                    "source_type": "llm_fallback",
                    "rna_cells": int(fallback_result[0].n_obs),
                    "rna_genes": int(fallback_result[0].n_vars),
                    "atac_cells": int(fallback_result[1].n_obs),
                    "atac_features": int(fallback_result[1].n_vars),
                }
            ]
            multiome_inputs = []
        else:
            advice_path, code_path = _write_fallback_guidance(
                raw_input_path=staged_input,
                output_dir=outdir,
                logger=logger,
                model_name=annotation_model,
                reason="No supported 10x multiome input was found.",
            )
            raise ValueError(
                "No supported 10x multiome input was found. "
                f"See {advice_path} and {code_path} for dataset-specific preprocessing guidance."
            )
    else:
        rna_parts = []
        atac_parts = []
        sample_rows = []
        try:
            for sample in sorted(multiome_inputs, key=lambda item: item.sample_key):
                adata_rna, adata_atac = _load_multiome_sample(sample)
                rna_parts.append(adata_rna)
                atac_parts.append(adata_atac)
                sample_rows.append(
                    {
                        "sample_key": sample.sample_key,
                        "sample_id": sample.sample_id,
                        "tissue": sample.tissue,
                        "source_type": sample.source_type,
                        "rna_cells": int(adata_rna.n_obs),
                        "rna_genes": int(adata_rna.n_vars),
                        "atac_cells": int(adata_atac.n_obs),
                        "atac_features": int(adata_atac.n_vars),
                    }
                )
        except Exception as exc:
            fallback_result = _try_llm_fallback_preprocessing(
                raw_input_path=staged_input,
                output_dir=outdir,
                logger=logger,
                model_name=annotation_model,
                reason=f"Could not parse a standard multiome input: {exc}",
            )
            if fallback_result is not None:
                rna_parts = [fallback_result[0]]
                atac_parts = [fallback_result[1]]
                sample_rows = [
                    {
                        "sample_key": "llm_fallback",
                        "sample_id": "llm_fallback",
                        "tissue": "unknown",
                        "source_type": "llm_fallback",
                        "rna_cells": int(fallback_result[0].n_obs),
                        "rna_genes": int(fallback_result[0].n_vars),
                        "atac_cells": int(fallback_result[1].n_obs),
                        "atac_features": int(fallback_result[1].n_vars),
                    }
                ]
            else:
                advice_path, code_path = _write_fallback_guidance(
                    raw_input_path=staged_input,
                    output_dir=outdir,
                    logger=logger,
                    model_name=annotation_model,
                    reason=f"Could not parse a standard multiome input: {exc}",
                )
                raise ValueError(
                    f"Could not parse the raw multiome input: {exc}. "
                    f"See {advice_path} and {code_path} for dataset-specific preprocessing guidance."
                ) from exc

    rna = ad.concat(rna_parts, join="outer", merge="same") if len(rna_parts) > 1 else rna_parts[0]
    atac = ad.concat(atac_parts, join="outer", merge="same") if len(atac_parts) > 1 else atac_parts[0]
    rna.var_names_make_unique()
    atac.var_names_make_unique()
    rna.var_names.name = None
    atac.var_names.name = None

    rna.var["mt"] = rna.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], inplace=True)
    pre_qc = _sample_qc_summary(rna, "rna_before_filter")

    atac.obs["total_counts"] = np.asarray(_ensure_csr(atac.X).sum(axis=1)).ravel()
    atac.obs["n_features_by_counts"] = _n_nonzero_by_row(atac.X)
    atac.obs["pct_counts_mt"] = 0.0

    rna_keep = (rna.obs["n_genes_by_counts"] >= min_genes) & (rna.obs["pct_counts_mt"] <= max_pct_mt)
    atac_keep = atac.obs["n_features_by_counts"] >= min_atac_features
    kept_names = set(map(str, rna.obs_names[rna_keep])).intersection(map(str, atac.obs_names[atac_keep]))
    if not kept_names:
        raise ValueError("No cells passed the combined RNA and ATAC QC filters. Try a lower ATAC threshold or inspect the raw inputs.")
    rna_order = [name for name in rna.obs_names if str(name) in kept_names]
    atac_order = [name for name in atac.obs_names if str(name) in kept_names]
    rna = rna[rna_order].copy()
    atac = atac[atac_order].copy()

    sc.pp.filter_genes(rna, min_cells=min_cells)
    sc.pp.filter_genes(atac, min_cells=min_cells)
    post_qc = _sample_qc_summary(rna, "rna_after_filter")

    rna.layers["counts"] = rna.X.copy()
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    rna.raw = rna.copy()
    batch_key = "sample_key" if "sample_key" in rna.obs.columns and rna.obs["sample_key"].nunique() > 1 else None
    sc.pp.highly_variable_genes(rna, n_top_genes=n_top_genes, flavor="seurat", batch_key=batch_key, subset=False)
    hvg_mask = rna.var["highly_variable"].fillna(False).to_numpy()
    if hvg_mask.sum() == 0:
        raise ValueError("No highly variable genes were selected from the RNA matrix.")
    rna_hvg = rna[:, hvg_mask].copy()
    sc.pp.scale(rna_hvg, max_value=10)
    sc.tl.pca(rna_hvg, svd_solver="arpack")
    sc.pp.neighbors(rna_hvg, n_neighbors=min(n_neighbors, max(2, rna_hvg.n_obs - 1)), n_pcs=min(n_pcs, rna_hvg.obsm["X_pca"].shape[1]))
    sc.tl.umap(rna_hvg)
    sc.tl.leiden(rna_hvg, resolution=leiden_resolution, key_added="leiden")
    rna.obs["leiden"] = rna_hvg.obs["leiden"].astype("category")
    rna.obsm["X_pca"] = rna_hvg.obsm["X_pca"]
    rna.obsm["X_umap"] = rna_hvg.obsm["X_umap"]
    rna.uns["neighbors"] = rna_hvg.uns["neighbors"]
    if "umap" in rna_hvg.uns:
        rna.uns["umap"] = rna_hvg.uns["umap"]
    sc.tl.rank_genes_groups(rna, groupby="leiden", method="wilcoxon", use_raw=True)
    rna_marker_df = _extract_marker_table(rna, top_n=marker_top_n)
    rna_marker_df["used_for_annotation"] = False
    for cluster in rna_marker_df["cluster"].astype(str).unique():
        mask = (rna_marker_df["cluster"].astype(str) == cluster) & (~rna_marker_df["is_linc_like"])
        rna_marker_df.loc[rna_marker_df.loc[mask].head(annotation_marker_top_n).index, "used_for_annotation"] = True

    annotation_notes = read_text(annotation_notes_path) if annotation_notes_path else ""
    rna_annotation_df = _annotate_clusters_with_llm(
        rna_marker_df,
        model_name=annotation_model,
        logger=logger,
        annotation_notes=(annotation_notes + "\nFocus on broad cell identity or major cell state labels.").strip(),
    )
    rna_label_map = dict(zip(rna_annotation_df["cluster_id"].astype(str), rna_annotation_df["cell_type"].astype(str)))
    rna.obs["cluster_cell_type"] = rna.obs["leiden"].astype(str).map(rna_label_map).fillna("unresolved").astype("category")
    rna.obs["cell_type"] = rna.obs["cluster_cell_type"].astype(str).astype("category")

    atac.layers["counts"] = atac.X.copy()
    sc.pp.normalize_total(atac, target_sum=1e4)
    sc.pp.log1p(atac)
    atac.raw = atac.copy()
    _run_atac_lsi(atac, n_components=max(n_pcs + 1, 10), n_neighbors=n_neighbors, leiden_resolution=leiden_resolution)

    shared_obs = sorted(set(map(str, atac.obs_names)).intersection(map(str, rna.obs_names)))
    atac.obs["paired_rna_label"] = "unresolved"
    if shared_obs:
        atac.obs.loc[shared_obs, "paired_rna_label"] = rna.obs.loc[shared_obs, "cluster_cell_type"].astype(str).values
    atac_annotation_df = _make_atac_annotation_table(atac, rna)
    atac_label_map = dict(zip(atac_annotation_df["cluster_id"].astype(str), atac_annotation_df["cell_type"].astype(str)))
    atac.obs["cluster_cell_type"] = atac.obs["leiden"].astype(str).map(atac_label_map).fillna("unresolved").astype("category")
    atac.obs["cell_type"] = atac.obs["cluster_cell_type"].astype(str).astype("category")

    try:
        gene_activity, gene_activity_names = _build_gene_activity_matrix(rna, atac)
        atac.obsm["gene_activity"] = gene_activity
        atac.uns["gene_activity_var_names"] = gene_activity_names
    except Exception as exc:
        logger.warning(f"Gene activity construction failed; continuing with peak-level ATAC only. Error: {exc}")

    sc.tl.rank_genes_groups(atac, groupby="leiden", method="wilcoxon", use_raw=True)
    atac_marker_df = _extract_marker_table(atac, top_n=marker_top_n)

    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rna_umap_path = figures_dir / "rna_umap_cluster_cell_type.png"
    atac_umap_path = figures_dir / "atac_umap_cluster_cell_type.png"
    _save_umap_figure(rna, "cluster_cell_type", rna_umap_path, "scRNA UMAP by cluster annotation")
    _save_umap_figure(atac, "cluster_cell_type", atac_umap_path, "scATAC UMAP by transferred annotation")

    rna_h5ad_path = outdir / "processed_rna.h5ad"
    atac_h5ad_path = outdir / "processed_atac.h5ad"
    rna_markers_path = outdir / "rna_cluster_markers.csv"
    atac_markers_path = outdir / "atac_cluster_markers.csv"
    rna_annotations_path = outdir / "rna_cluster_annotations.csv"
    atac_annotations_path = outdir / "atac_cluster_annotations.csv"
    qc_summary_path = outdir / "qc_summary.txt"
    manifest_path = outdir / "prep_manifest.json"

    rna.write_h5ad(rna_h5ad_path)
    atac.write_h5ad(atac_h5ad_path)
    rna_marker_df.to_csv(rna_markers_path, index=False)
    atac_marker_df.to_csv(atac_markers_path, index=False)
    rna_annotation_df.to_csv(rna_annotations_path, index=False)
    atac_annotation_df.to_csv(atac_annotations_path, index=False)

    sample_table = pd.DataFrame(sample_rows)
    qc_lines = [
        "Sample table",
        sample_table.to_string(index=False) if not sample_table.empty else "No samples found.",
        "",
        "RNA QC before filtering",
        pre_qc.to_string(index=False) if not pre_qc.empty else "No RNA cells before filtering.",
        "",
        "RNA QC after filtering",
        post_qc.to_string(index=False) if not post_qc.empty else "No RNA cells after filtering.",
        "",
        f"ATAC cells after filtering: {atac.n_obs}",
        f"ATAC features after filtering: {atac.n_vars}",
        f"Shared cells after filtering: {len(shared_obs)}",
        f"Minimum ATAC detected features threshold: {min_atac_features}",
        f"Gene activity available: {'gene_activity' in atac.obsm and 'gene_activity_var_names' in atac.uns}",
    ]
    qc_summary_path.write_text("\n".join(qc_lines), encoding="utf-8")

    manifest = {
        "raw_input_path": str(Path(raw_input_path).resolve()),
        "rna_h5ad_path": str(rna_h5ad_path),
        "atac_h5ad_path": str(atac_h5ad_path),
        "rna_cells_after_filter": int(rna.n_obs),
        "rna_genes_after_filter": int(rna.n_vars),
        "atac_cells_after_filter": int(atac.n_obs),
        "atac_features_after_filter": int(atac.n_vars),
        "shared_cells_after_filter": int(len(shared_obs)),
        "gene_activity_available": bool("gene_activity" in atac.obsm and "gene_activity_var_names" in atac.uns),
        "gene_activity_features": int(len(atac.uns.get("gene_activity_var_names", []))),
        "rna_clusters": int(rna.obs["leiden"].nunique()),
        "atac_clusters": int(atac.obs["leiden"].nunique()),
        "samples": sample_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"scRNA + scATAC preparation complete. Outputs written to {outdir}")

    return PreparationResult(
        output_dir=outdir,
        rna_h5ad_path=rna_h5ad_path,
        atac_h5ad_path=atac_h5ad_path,
        rna_markers_path=rna_markers_path,
        atac_markers_path=atac_markers_path,
        rna_annotations_path=rna_annotations_path,
        atac_annotations_path=atac_annotations_path,
        qc_summary_path=qc_summary_path,
        manifest_path=manifest_path,
        rna_umap_path=rna_umap_path,
        atac_umap_path=atac_umap_path,
    )
