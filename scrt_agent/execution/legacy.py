"""Notebook execution backend for scMAA."""

from __future__ import annotations

from pathlib import Path
from queue import Empty
import re
import shutil
import time

import litellm
import nbformat as nbf
import openai
from jupyter_client import KernelManager
from nbformat.v4 import new_code_cell, new_markdown_cell, new_output

from ..hypothesis import AnalysisPlan
from ..research import ResearchLedger
from ..utils import get_documentation, summarize_notebook_cells, truncate_text
from ..validator import DatasetValidator


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python") :]
    elif text.startswith("```"):
        text = text[len("```") :]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


class LegacyNotebookExecutor:
    """Notebook executor using a persistent Jupyter kernel."""

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
        tcr_summary: str,
        joint_summary: str,
        validation_summary: str,
        standard_baseline_summary: str,
        context_summary: str,
        logger,
        rna_h5ad_path: str,
        tcr_path: str,
        output_dir: str | Path,
        analysis_name: str,
        max_iterations: int = 6,
        max_fix_attempts: int = 3,
        use_VLM: bool = True,
        use_documentation: bool = True,
    ) -> None:
        self.hypothesis_generator = hypothesis_generator
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.vision_model = vision_model
        self.prompt_dir = Path(prompt_dir)
        self.coding_guidelines = coding_guidelines
        self.coding_system_prompt = coding_system_prompt
        self.rna_summary = rna_summary
        self.tcr_summary = tcr_summary
        self.joint_summary = joint_summary
        self.validation_summary = validation_summary
        self.standard_baseline_summary = standard_baseline_summary
        self.context_summary = context_summary
        self.logger = logger
        self.rna_h5ad_path = str(Path(rna_h5ad_path))
        self.tcr_path = str(Path(tcr_path))
        self.output_dir = Path(output_dir)
        self.analysis_name = analysis_name
        self.project_root = Path(__file__).resolve().parents[2]
        self.max_iterations = max_iterations
        self.max_fix_attempts = max_fix_attempts
        self.use_VLM = use_VLM
        self.use_documentation = use_documentation
        self.kernel_manager: KernelManager | None = None
        self.kernel_client = None
        self.code_memory: list[str] = []
        self.code_memory_size = 5
        self.vision_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.step_validator = DatasetValidator()

    def _read_prompt(self, name: str) -> str:
        return (self.prompt_dir / name).read_text(encoding="utf-8")

    def update_code_memory(self, notebook_cells: list) -> None:
        code_cells = [cell.source for cell in notebook_cells if getattr(cell, "cell_type", "") == "code"]
        self.code_memory = code_cells[-self.code_memory_size :]

    def start_persistent_kernel(self) -> None:
        self.kernel_manager = KernelManager(kernel_name="python3")
        self.kernel_manager.start_kernel()
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()
        self.kernel_client.wait_for_ready(timeout=60)
        self.logger.info("Persistent Jupyter kernel started.")

    def stop_persistent_kernel(self) -> None:
        try:
            if self.kernel_client is not None:
                self.kernel_client.stop_channels()
            if self.kernel_manager is not None:
                self.kernel_manager.shutdown_kernel(now=True)
        finally:
            self.kernel_client = None
            self.kernel_manager = None

    def create_initial_notebook(self, analysis, research_ledger: ResearchLedger) -> nbf.NotebookNode:
        notebook = nbf.v4.new_notebook()
        notebook.cells.append(new_markdown_cell("# scMAA Analysis"))
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

from scrt_agent.notebook_tools import (
    assign_clone_type_labels,
    clone_expansion_table,
    clone_type_distribution_table,
    expanded_clone_tissue_de,
    expression_frame,
    ensure_obs_column,
    ensure_obs_columns,
    infer_tumor_like_tissues,
    infer_primary_metastasis_tissues,
    paired_tcr_subset,
    plot_de_barplot,
    plot_de_heatmap,
    plot_tissue_embedding,
    print_clone_expansion_table,
    proportion_table,
    recluster_and_annotate_t_cells,
    resolve_gene_names,
    safe_rank_genes_groups,
    summarize_de_pathways,
    t_cell_subset,
    t_cell_cluster_marker_summary,
    tissue_stratified_expansion_de,
    tumor_like_subset,
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
TCR_PATH = r'''{self.tcr_path}'''

def _load_tcr_table(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {{".tsv", ".txt"}}:
        return pd.read_csv(path, sep="\\t")
    if suffix in {{".gz", ".bz2"}}:
        if path.name.endswith(".csv.gz"):
            return pd.read_csv(path)
        return pd.read_csv(path, sep="\\t")
    raise ValueError(f"Unsupported TCR table format: {{path}}")

def _normalize_barcode(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    return text.split("-")[0]

def _normalize_barcode_exact(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    return text

def _normalize_sample(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return text if text else np.nan

def _make_merge_key(barcode, sample=None, use_core=False):
    barcode_value = _normalize_barcode(barcode) if use_core else _normalize_barcode_exact(barcode)
    if pd.isna(barcode_value) or not str(barcode_value).strip():
        return np.nan
    sample_value = _normalize_sample(sample)
    if pd.isna(sample_value):
        return str(barcode_value)
    return f"{{sample_value}}::{{barcode_value}}"

def _coerce_bool(value):
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {{"true", "t", "1", "yes", "y", "productive", "high"}}

def _prepare_tcr_table(df):
    aliases = {{
        "barcode": ["barcode", "cell_id", "cell_barcode"],
        "sample_id": ["sample", "sample_id", "orig.ident", "donor", "patient", "library_id"],
        "sample_key": ["sample_key", "sample_name"],
        "clonotype_id": ["clonotype_id", "raw_clonotype_id", "clone_id", "clonotype"],
        "chain": ["chain", "locus"],
        "cdr3": ["cdr3", "cdr3_aa", "cdr3s_aa", "cdr3_nt"],
        "v_gene": ["v_gene", "v_call", "v_segment", "trav", "trbv"],
        "j_gene": ["j_gene", "j_call", "j_segment", "traj", "trbj"],
        "productive": ["productive", "high_confidence", "is_productive"],
        "reads": ["reads", "umis", "consensus_count", "duplicate_count"],
    }}
    out = df.copy()
    lower_to_original = {{str(col).lower(): col for col in out.columns}}
    for target, candidates in aliases.items():
        if target in out.columns:
            continue
        for candidate in candidates:
            if candidate.lower() in lower_to_original:
                out[target] = out[lower_to_original[candidate.lower()]]
                break
    if "barcode" not in out.columns:
        raise ValueError("TCR table must contain a barcode-like column.")
    out["barcode"] = out["barcode"].astype(str)
    out["barcode_core"] = out["barcode"].map(_normalize_barcode)
    if "productive" in out.columns:
        out["productive"] = out["productive"].map(_coerce_bool)
    else:
        out["productive"] = False
    return out

def _sample_scope_column(df):
    for column in ("sample_key", "sample_id"):
        if column in df.columns:
            return column
    return None

def _sample_scope_column_obs(df):
    for column in ("sample_key", "sample_id", "sample", "orig.ident", "donor", "patient"):
        if column in df.columns:
            return column
    return None

def _needs_sample_prefixed_clonotypes(df):
    if "clonotype_id" not in df.columns:
        return False, None
    sample_col = _sample_scope_column(df)
    if sample_col is None:
        return False, None
    scoped = df.loc[df["clonotype_id"].notna(), [sample_col, "clonotype_id"]].drop_duplicates()
    if scoped.empty:
        return False, sample_col
    spread = scoped.groupby("clonotype_id")[sample_col].nunique()
    risky = spread[spread > 1]
    if risky.empty:
        return False, sample_col
    raw_like = risky.index.to_series().astype(str).str.fullmatch(r"clonotype\\d+", case=False, na=False)
    return float(raw_like.mean()) >= 0.5, sample_col

def _join_unique(series):
    values = [str(v) for v in series if pd.notna(v) and str(v).strip()]
    return "|".join(sorted(set(values))) if values else np.nan

def _aggregate_tcr_by_column(df, column):
    grouped = df.groupby(column, dropna=False)
    agg = pd.DataFrame(index=grouped.size().index)
    agg["clonotype_id"] = grouped["clonotype_id"].agg(lambda s: next((v for v in s if pd.notna(v)), np.nan)) if "clonotype_id" in df.columns else np.nan
    agg["chain"] = grouped["chain"].agg(_join_unique) if "chain" in df.columns else np.nan
    agg["cdr3"] = grouped["cdr3"].agg(_join_unique) if "cdr3" in df.columns else np.nan
    agg["v_gene"] = grouped["v_gene"].agg(_join_unique) if "v_gene" in df.columns else np.nan
    agg["j_gene"] = grouped["j_gene"].agg(_join_unique) if "j_gene" in df.columns else np.nan
    agg["productive_any"] = grouped["productive"].agg("max")
    agg["tcr_chain_count"] = grouped.size()
    if "reads" in df.columns:
        agg["tcr_reads"] = pd.to_numeric(df["reads"], errors="coerce").groupby(df[column]).sum(min_count=1)
    return agg

print("Loading RNA and TCR inputs...")
adata_rna = sc.read_h5ad(RNA_H5AD_PATH)
tcr_df = _prepare_tcr_table(_load_tcr_table(TCR_PATH))
needs_prefix, sample_scope = _needs_sample_prefixed_clonotypes(tcr_df)
clonotype_scope = "as_provided"
if needs_prefix:
    mask = tcr_df["clonotype_id"].notna()
    tcr_df.loc[mask, "clonotype_id"] = tcr_df.loc[mask, sample_scope].astype(str) + ":" + tcr_df.loc[mask, "clonotype_id"].astype(str)
    clonotype_scope = f"prefixed_by_{{sample_scope}}"

if "barcode" in adata_rna.obs.columns:
    adata_rna.obs["barcode"] = adata_rna.obs["barcode"].astype(str)
else:
    adata_rna.obs["barcode"] = adata_rna.obs_names.astype(str)
adata_rna.obs["barcode_exact"] = adata_rna.obs["barcode"].map(_normalize_barcode_exact)
adata_rna.obs["barcode_core"] = adata_rna.obs["barcode"].map(_normalize_barcode)

rna_sample_scope = _sample_scope_column_obs(adata_rna.obs)
tcr_sample_scope = _sample_scope_column(tcr_df)
adata_rna.obs["sample_merge_key"] = adata_rna.obs.apply(
    lambda row: _make_merge_key(row["barcode"], row[rna_sample_scope] if rna_sample_scope else np.nan, use_core=False),
    axis=1,
)
adata_rna.obs["sample_merge_key_core"] = adata_rna.obs.apply(
    lambda row: _make_merge_key(row["barcode"], row[rna_sample_scope] if rna_sample_scope else np.nan, use_core=True),
    axis=1,
)
if tcr_sample_scope:
    tcr_df["sample_merge_key"] = [
        _make_merge_key(barcode, sample, use_core=False)
        for barcode, sample in zip(tcr_df["barcode"], tcr_df[tcr_sample_scope])
    ]
    tcr_df["sample_merge_key_core"] = [
        _make_merge_key(barcode, sample, use_core=True)
        for barcode, sample in zip(tcr_df["barcode"], tcr_df[tcr_sample_scope])
    ]

tcr_cell_exact = _aggregate_tcr_by_column(tcr_df, "barcode")
tcr_cell_core = _aggregate_tcr_by_column(tcr_df, "barcode_core")
tcr_cell_sample_exact = _aggregate_tcr_by_column(tcr_df, "sample_merge_key") if "sample_merge_key" in tcr_df.columns else pd.DataFrame()
tcr_cell_sample_core = _aggregate_tcr_by_column(tcr_df, "sample_merge_key_core") if "sample_merge_key_core" in tcr_df.columns else pd.DataFrame()

exact_overlap = int(adata_rna.obs["barcode_exact"].isin(tcr_cell_exact.index).sum())
core_overlap = int(adata_rna.obs["barcode_core"].isin(tcr_cell_core.index).sum())
sample_exact_overlap = int(adata_rna.obs["sample_merge_key"].isin(tcr_cell_sample_exact.index).sum()) if not tcr_cell_sample_exact.empty else 0
sample_core_overlap = int(adata_rna.obs["sample_merge_key_core"].isin(tcr_cell_sample_core.index).sum()) if not tcr_cell_sample_core.empty else 0
overlap_modes = {{
    "sample_exact": sample_exact_overlap,
    "sample_barcode_core": sample_core_overlap,
    "exact": exact_overlap,
    "barcode_core": core_overlap,
}}
merge_mode = max(overlap_modes, key=overlap_modes.get)

if merge_mode == "sample_exact":
    adata_rna.obs = adata_rna.obs.join(tcr_cell_sample_exact, on="sample_merge_key")
elif merge_mode == "sample_barcode_core":
    adata_rna.obs = adata_rna.obs.join(tcr_cell_sample_core, on="sample_merge_key_core")
elif merge_mode == "exact":
    adata_rna.obs = adata_rna.obs.join(tcr_cell_exact, on="barcode_exact")
else:
    adata_rna.obs = adata_rna.obs.join(tcr_cell_core, on="barcode_core")

adata_rna.obs["has_tcr"] = adata_rna.obs["clonotype_id"].notna()
clone_sizes = adata_rna.obs.loc[adata_rna.obs["has_tcr"], "clonotype_id"].value_counts()
adata_rna.obs["clone_size"] = adata_rna.obs["clonotype_id"].map(clone_sizes).fillna(0).astype(int)
adata_rna.obs["expanded_clone"] = adata_rna.obs["clone_size"] >= 3
adata_rna.obs["singleton_clone"] = adata_rna.obs["clone_size"] == 1
adata_rna.obs["small_clone"] = adata_rna.obs["clone_size"].between(2, 4)
ensure_obs_columns(adata_rna, ["sample_id", "tissue", "sample_key"], fill_value="Unknown", as_category=True)
if "cell_type" not in adata_rna.obs.columns:
    for alias in ("cluster_cell_type", "celltype", "celltypes", "annotation", "annotated_cell_type"):
        if alias in adata_rna.obs.columns:
            adata_rna.obs["cell_type"] = adata_rna.obs[alias].astype(str)
            print(f"Created cell_type alias from {{alias}}")
            break
ensure_obs_column(adata_rna, "cell_type", fill_value="Unknown", as_category=True)

print(f"RNA shape: {{adata_rna.n_obs}} x {{adata_rna.n_vars}}")
print(f"TCR rows: {{len(tcr_df)}}")
print(f"TCR unique barcodes: {{tcr_df['barcode'].nunique()}}")
print(f"Exact barcode overlap: {{exact_overlap}}")
print(f"Core barcode overlap: {{core_overlap}}")
print(f"Sample-aware exact overlap: {{sample_exact_overlap}}")
print(f"Sample-aware core overlap: {{sample_core_overlap}}")
print(f"Chosen merge mode: {{merge_mode}}")
print(f"Clonotype scope: {{clonotype_scope}}")
print(f"Cells with TCR annotations after merge: {{int(adata_rna.obs['has_tcr'].sum())}}")
print(f"Expanded-clone fraction among TCR+ cells: {{float(adata_rna.obs.loc[adata_rna.obs['has_tcr'], 'expanded_clone'].mean()):.3f}}")
print("Notebook helper functions available: paired_tcr_subset, infer_tumor_like_tissues, infer_primary_metastasis_tissues, tumor_like_subset, resolve_gene_names, expression_frame, print_clone_expansion_table, safe_rank_genes_groups, tissue_stratified_expansion_de, expanded_clone_tissue_de, plot_de_barplot, plot_de_heatmap, plot_tissue_embedding, summarize_de_pathways")
"""
        notebook.cells.append(new_code_cell(setup_code))
        notebook.cells.append(
            new_markdown_cell(
                "## Standard Baseline Analysis\n\n"
                "The agent must first review a fixed baseline analysis before proposing any innovative hypothesis.\n\n"
                f"{self.standard_baseline_summary}"
            )
        )
        baseline_code = """# Deterministic baseline analysis before hypothesis-driven exploration
group_col = "cell_type" if "cell_type" in adata_rna.obs.columns else ("cluster_cell_type" if "cluster_cell_type" in adata_rna.obs.columns else "leiden")
tissue_col = "tissue" if "tissue" in adata_rna.obs.columns else ("sample_key" if "sample_key" in adata_rna.obs.columns else "sample_id")
paired_adata = paired_tcr_subset(adata_rna)
assign_clone_type_labels(paired_adata)

# 1. Global scRNA embedding with current annotations
if "X_umap" in adata_rna.obsm:
    sc.pl.umap(adata_rna, color=group_col, title="scRNA UMAP with annotations", frameon=False)

# 2. Global cell-type composition across tissues
global_prop = proportion_table(adata_rna, group_col=group_col, tissue_col=tissue_col, normalize="index")
if not global_prop.empty:
    ax = global_prop.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="tab20")
    ax.set_title("Cell-type composition across tissues")
    ax.set_ylabel("Fraction")
    ax.set_xlabel("")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.show()

# 3. T-cell subset reclustering / LLM-guided annotation
t_adata, t_marker_df, t_annotation_df = recluster_and_annotate_t_cells(
    paired_adata,
    group_col=group_col,
    model_name="gpt-4o",
)
t_group_col = "tcell_cluster_cell_type" if "tcell_cluster_cell_type" in t_adata.obs.columns else group_col
sc.pl.umap(
    t_adata,
    color=["tcell_leiden", t_group_col],
    title=["T-cell reclustering", "LLM-defined T-cell annotation"],
    frameon=False,
)
print("T-cell cluster marker summary (top non-lincRNA markers per cluster):")
print(t_cell_cluster_marker_summary(t_marker_df, top_n=50))
if not t_annotation_df.empty:
    print("T-cell cluster annotations:")
    print(t_annotation_df[["cluster_id", "cell_type", "confidence"]].to_string(index=False))

# 4. T-cell subtype composition across tissues
t_prop = proportion_table(t_adata, group_col=t_group_col, tissue_col=tissue_col, normalize="index")
if not t_prop.empty:
    ax = t_prop.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="tab20")
    ax.set_title("T-cell subtype composition across tissues")
    ax.set_ylabel("Fraction")
    ax.set_xlabel("")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.show()

# 5. cloneType UMAP within T cells
if "X_umap" in t_adata.obsm:
    sc.pl.umap(t_adata, color=[t_group_col, "cloneType"], title=["T-cell annotation", "T-cell cloneType"], frameon=False)

# 6. cloneType composition across T-cell subtypes
clone_mix = clone_type_distribution_table(t_adata, group_col=t_group_col, clone_type_col="cloneType")
if not clone_mix.empty:
    ax = clone_mix.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="viridis")
    ax.set_title("cloneType composition across T-cell subtypes")
    ax.set_ylabel("Fraction")
    ax.set_xlabel("")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.show()

# 7. Baseline printed priorities for downstream hypothesis generation
baseline_priority = (
    t_adata.obs.groupby(t_group_col, observed=False)["expanded_clone"]
    .agg(paired_cells="size", expanded_fraction="mean")
    .sort_values(["expanded_fraction", "paired_cells"], ascending=[False, False])
)
print("Baseline T-cell priority table:")
print(baseline_priority.head(10).to_string())
"""
        notebook.cells.append(new_code_cell(baseline_code))
        return notebook

    def _save_notebook(self, notebook: nbf.NotebookNode, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            nbf.write(notebook, handle)

    def _backup_existing_notebook(self, path: Path) -> Path | None:
        if not path.exists():
            return None
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.stem}_{timestamp}.bak{path.suffix}")
        shutil.copy2(path, backup_path)
        return backup_path

    def _get_last_code_cell(self, notebook: nbf.NotebookNode):
        for cell in reversed(notebook.cells):
            if cell.cell_type == "code":
                return cell
        return None

    def _get_code_cells(self, notebook: nbf.NotebookNode) -> list:
        return [cell for cell in notebook.cells if getattr(cell, "cell_type", "") == "code"]

    def _run_code_cell(self, cell) -> tuple[bool, str | None]:
        if self.kernel_client is None:
            raise RuntimeError("Kernel has not been started.")

        msg_id = self.kernel_client.execute(cell.source)
        outputs = []
        error_text = None
        start_time = time.time()
        last_message_time = start_time
        inactivity_timeout = 300
        total_timeout = 1800

        while True:
            try:
                msg = self.kernel_client.get_iopub_msg(timeout=30)
            except Empty:
                now = time.time()
                if now - start_time > total_timeout:
                    error_text = f"TimeoutError: code cell exceeded {total_timeout} seconds of total runtime"
                    outputs.append(
                        new_output(
                            "error",
                            ename="TimeoutError",
                            evalue=f"code cell exceeded {total_timeout} seconds of total runtime",
                            traceback=[],
                        )
                    )
                    break
                if now - last_message_time > inactivity_timeout:
                    self.logger.warning(
                        f"No notebook output for {inactivity_timeout} seconds; continuing to wait for code cell completion."
                    )
                    last_message_time = now
                continue
            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["msg_type"]
            content = msg["content"]
            last_message_time = time.time()

            if msg_type == "status" and content.get("execution_state") == "idle":
                break
            if msg_type == "stream":
                outputs.append(new_output("stream", name=content["name"], text=content["text"]))
            elif msg_type == "execute_result":
                outputs.append(
                    new_output(
                        "execute_result",
                        data=content["data"],
                        execution_count=content.get("execution_count"),
                    )
                )
            elif msg_type == "display_data":
                outputs.append(new_output("display_data", data=content["data"], metadata=content.get("metadata", {})))
            elif msg_type == "error":
                outputs.append(
                    new_output(
                        "error",
                        ename=content["ename"],
                        evalue=content["evalue"],
                        traceback=content["traceback"],
                    )
                )
                error_text = f"{content['ename']}: {content['evalue']}"

        cell.outputs = outputs
        return error_text is None, error_text

    def run_last_code_cell(self, notebook: nbf.NotebookNode) -> tuple[bool, str | None]:
        last_code_cell = self._get_last_code_cell(notebook)
        if last_code_cell is None:
            raise ValueError("No code cell found to execute.")
        return self._run_code_cell(last_code_cell)

    def _collect_text_output(self, cell) -> str:
        parts: list[str] = []
        for output in getattr(cell, "outputs", []):
            output_type = output.get("output_type")
            if output_type == "stream":
                parts.append(str(output.get("text", "")))
            elif output_type == "execute_result":
                parts.append(str(output.get("data", {}).get("text/plain", "")))
            elif output_type == "display_data":
                data = output.get("data", {})
                if "text/plain" in data:
                    parts.append(str(data.get("text/plain", "")))
            elif output_type == "error":
                parts.append(f"{output.get('ename', '')}: {output.get('evalue', '')}")
        return "\n".join(part for part in parts if part).strip()

    def _collect_image_outputs(self, cell) -> list[str]:
        images: list[str] = []
        for output in getattr(cell, "outputs", []):
            if output.get("output_type") != "display_data":
                continue
            png = output.get("data", {}).get("image/png")
            if png:
                images.append(png.split(",")[-1] if "," in png else png)
        return images

    def _plan_item_tokens(self, text: str) -> set[str]:
        lowered = (text or "").lower()
        tokens: set[str] = set()
        for label, variants in self._plan_signal_labels().items():
            if any(variant in lowered for variant in variants):
                tokens.add(label)
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{3,}", lowered):
            if token in {"with", "that", "this", "from", "using", "into", "then", "perform", "analysis", "visualize", "results", "known", "novel", "subset", "study", "identify", "compare"}:
                continue
            tokens.add(token)
        return tokens

    def _plan_signal_labels(self) -> dict[str, tuple[str, ...]]:
        return {
            "pseudotime": ("pseudotime", "trajectory", "diffmap", "dpt", "palantir", "slingshot", "拟时序", "轨迹"),
            "heatmap": ("heatmap", "热图"),
            "barplot": ("barplot", "bar plot", "bar-chart", "bar chart", "柱状图"),
            "differential_expression": ("differential", "expression", "deg", "rank_genes_groups", "safe_rank_genes_groups", "差异表达", "差异基因"),
            "pathway_enrichment": ("pathway", "enrich", "enrichment", "gsea", "reactome", "kegg", "hallmark", "go:", "gene ontology", "msigdb"),
            "cellphonedb": ("cellphonedb", "cell phone db", "cellphone db", "ligand", "receptor", "communication", "通讯"),
            "paired_subset": ("paired_tcr_subset", "paired", "rna-tcr"),
            "tumor_subset": ("tumor_like_subset", "tumor-like", "metastasis", "primary_focus", "btc"),
            "clone": ("clonotype", "clone", "expansion", "sharing", "overlap"),
            "vj_usage": ("v gene", "j gene", "trbv", "trav", "vj"),
            "t_cell": ("t cell", "treg", "cd8", "cd4"),
            "cluster": ("leiden", "cluster_cell_type", "cluster label"),
            "visualization": ("plot", "visualize", "figure", "umap", "sc.pl", "plt."),
            "resolve_gene_names": ("resolve_gene_names", "resolved genes", "gene name"),
        }

    def _plan_item_requirements(self, item: str) -> tuple[set[str], set[str], set[str]]:
        item_tokens = self._plan_item_tokens(item)
        required_all: set[str] = set()
        required_any: set[str] = set()

        for label in (
            "pseudotime",
            "differential_expression",
            "pathway_enrichment",
            "cellphonedb",
            "paired_subset",
            "tumor_subset",
            "clone",
            "vj_usage",
            "t_cell",
            "cluster",
            "visualization",
            "resolve_gene_names",
        ):
            if label in item_tokens:
                required_all.add(label)

        if "heatmap" in item_tokens and "barplot" in item_tokens:
            required_any.update({"heatmap", "barplot"})
        else:
            if "heatmap" in item_tokens:
                required_all.add("heatmap")
            if "barplot" in item_tokens:
                required_all.add("barplot")

        return item_tokens, required_all, required_any

    def _execution_evidence(self, notebook: nbf.NotebookNode) -> dict[str, object]:
        code_parts: list[str] = []
        output_parts: list[str] = []
        image_count = 0
        code_index = 0
        for cell in notebook.cells:
            if cell.get("cell_type") != "code":
                continue
            code_index += 1
            if code_index == 1:
                # Skip the initial setup cell so helper imports and boilerplate do not
                # falsely count as completed approved-plan items.
                continue
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            if str(source).strip():
                code_parts.append(str(source))
            for output in cell.get("outputs", []):
                output_type = output.get("output_type")
                if output_type == "stream":
                    text = output.get("text", "")
                    if isinstance(text, list):
                        text = "".join(text)
                    if str(text).strip():
                        output_parts.append(str(text))
                elif output_type in {"execute_result", "display_data"}:
                    text = output.get("data", {}).get("text/plain", "")
                    if isinstance(text, list):
                        text = "".join(text)
                    if str(text).strip():
                        output_parts.append(str(text))
                    if output_type == "display_data" and output.get("data", {}).get("image/png"):
                        image_count += 1
                        output_parts.append("[image_output]")
                elif output_type == "error":
                    output_parts.append(f"{output.get('ename', '')}: {output.get('evalue', '')}")
        combined_text = "\n".join(code_parts + output_parts)
        output_text = "\n".join(output_parts)
        return {
            "combined_text": combined_text,
            "output_text": output_text,
            "combined_tokens": self._plan_item_tokens(combined_text.lower()),
            "image_count": image_count,
        }

    def _plan_item_completed(self, item: str, evidence: dict[str, object]) -> bool:
        item_tokens, required_all, required_any = self._plan_item_requirements(item)
        combined_tokens = evidence["combined_tokens"]
        if not isinstance(combined_tokens, set):
            return False
        if required_all and not required_all.issubset(combined_tokens):
            return False
        if required_any and not (required_any & combined_tokens):
            return False
        if not required_all and not required_any:
            return bool(item_tokens and (item_tokens & combined_tokens))

        evidence_labels = required_all | required_any
        combined_text = str(evidence.get("combined_text", "")).lower()
        output_text = str(evidence.get("output_text", "")).lower()
        image_count = int(evidence.get("image_count", 0) or 0)

        if evidence_labels & {"visualization", "heatmap", "barplot"}:
            figure_markers = ("[image_output]", ".png", ".pdf", "savefig", "saved figure", "figure saved", "publication_figure")
            if image_count <= 0 and not any(marker in combined_text for marker in figure_markers):
                return False

        if "pathway_enrichment" in evidence_labels:
            pathway_markers = ("pathway", "reactome", "kegg", "hallmark", "enrich", "enrichment", "gsea", "gene ontology", "go:")
            if not any(marker in output_text for marker in pathway_markers):
                return False

        return True

    def _pending_plan_items(self, approved_plan_items: list[str], notebook: nbf.NotebookNode) -> tuple[list[str], list[str]]:
        completed: list[str] = []
        pending: list[str] = []
        evidence = self._execution_evidence(notebook)
        for item in approved_plan_items:
            if self._plan_item_completed(item, evidence):
                completed.append(item)
            else:
                pending.append(item)
        return pending, completed

    def _analysis_advances_pending_items(self, analysis, pending_items: list[str]) -> bool:
        if not pending_items:
            return True
        code_text = "\n".join(
            [
                analysis.code_description,
                analysis.first_step_code,
            ]
        ).lower()
        combined = "\n".join(
            [
                analysis.priority_question,
                analysis.evidence_goal,
                analysis.code_description,
                analysis.first_step_code,
                "\n".join(analysis.analysis_plan),
                analysis.summary,
            ]
        ).lower()
        code_tokens = self._plan_item_tokens(code_text)
        combined_tokens = self._plan_item_tokens(combined)
        for item in pending_items:
            item_tokens, required_all, required_any = self._plan_item_requirements(item)
            proposal_tokens = combined_tokens
            if {"visualization", "heatmap", "barplot", "pathway_enrichment"} & (required_all | required_any):
                proposal_tokens = code_tokens
            if required_all and not required_all.issubset(proposal_tokens):
                continue
            if required_any and not (required_any & proposal_tokens):
                continue
            if not required_all and not required_any and not (item_tokens and (item_tokens & proposal_tokens)):
                continue
            if "pathway_enrichment" in (required_all | required_any):
                pathway_markers = ("pathway", "reactome", "kegg", "hallmark", "enrich", "enrichment", "gsea", "gene ontology", "go:")
                if not any(marker in code_text for marker in pathway_markers):
                    continue
            if {"visualization", "heatmap", "barplot"} & (required_all | required_any):
                figure_markers = ("plt.show", "savefig", "heatmap", "barplot", "bar plot", "sc.pl", "umap", "pca", "spatial")
                if not any(marker in code_text for marker in figure_markers):
                    continue
                return True
        return False

    def _plan_item_labels(self, items: list[str]) -> set[str]:
        labels: set[str] = set()
        for item in items:
            _, required_all, required_any = self._plan_item_requirements(item)
            labels.update(required_all)
            labels.update(required_any)
        return labels

    def _build_plan_completion_fallback(
        self,
        current_analysis,
        pending_items: list[str],
        completed_items: list[str],
    ) -> AnalysisPlan | None:
        pending_labels = self._plan_item_labels(pending_items)
        completed_labels = self._plan_item_labels(completed_items)
        needs_figure = bool({"visualization", "heatmap", "barplot"} & pending_labels)
        needs_pathway = "pathway_enrichment" in pending_labels
        if not needs_figure and not needs_pathway:
            return None
        if "paired_subset" not in completed_labels and "differential_expression" not in completed_labels:
            return None

        code = f"""reference_tissue, case_tissue = infer_primary_metastasis_tissues(adata_rna, tissue_col="tissue")
embedding_basis, embedding_df, embedding_ax = plot_tissue_embedding(
    adata_rna,
    tissue_col="tissue",
    expansion_col="expanded_clone",
    tissues=[reference_tissue, case_tissue],
    paired_only=True,
    basis="auto",
    title=f"Expanded clonotypes {{reference_tissue}} vs {{case_tissue}}",
)
fig_dir = Path(PROJECT_ROOT) / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
embedding_path = fig_dir / f"scrt_expanded_clone_tissue_{{embedding_basis}}.png"
embedding_ax.figure.savefig(embedding_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close(embedding_ax.figure)
print(f"Saved figure: {{embedding_path}}")

de_subset, de_results = expanded_clone_tissue_de(
    adata_rna,
    tissue_col="tissue",
    expansion_col="expanded_clone",
    tissues=[reference_tissue, case_tissue],
    case_tissue=case_tissue,
    reference_tissue=reference_tissue,
    paired_only=True,
    min_cells_per_group=20,
    top_n=40,
)

display_cols = ["names", "scores", "logfoldchanges", "pvals_adj"]
print("Top DE genes for expanded clonotypes in tissue contrast:")
print(de_results[display_cols].head(12).to_string(index=False))

barplot_ax = plot_de_barplot(
    de_results,
    n=12,
    title=f"Expanded clonotypes: {{case_tissue}} vs {{reference_tissue}}",
)
barplot_path = fig_dir / "scrt_expanded_clone_tissue_barplot.png"
barplot_ax.figure.savefig(barplot_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close(barplot_ax.figure)
print(f"Saved figure: {{barplot_path}}")

heatmap_genes = de_results["names"].dropna().astype(str).head(14).tolist()
heatmap_matrix, heatmap_ax = plot_de_heatmap(
    adata_rna,
    heatmap_genes,
    tissue_col="tissue",
    tissues=[reference_tissue, case_tissue],
    expansion_col="expanded_clone",
    paired_only=True,
    title=f"Expanded clonotypes heatmap: {{reference_tissue}} vs {{case_tissue}}",
)
heatmap_path = fig_dir / "scrt_expanded_clone_tissue_heatmap.png"
heatmap_ax.figure.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close(heatmap_ax.figure)
print(f"Saved figure: {{heatmap_path}}")

pathway_tables = summarize_de_pathways(
    de_results,
    case_label=case_tissue,
    reference_label=reference_tissue,
    n_case=15,
    n_reference=15,
)
"""
        return AnalysisPlan(
            hypothesis=current_analysis.hypothesis,
            analysis_type=current_analysis.analysis_type,
            priority_question=current_analysis.priority_question,
            evidence_goal=(
                "Produce the required DE visual summaries and a pathway-level interpretation "
                "for expanded clonotypes between primary and metastatic tissues."
            ),
            decision_rationale=(
                "The approved plan is already narrowed to visualization and functional interpretation, "
                "so the next step should execute those deliverables directly instead of revisiting setup checks."
            ),
            validation_checks=[
                "Use only paired TCR-positive cells annotated as expanded clonotypes.",
                "Contrast inferred primary and metastasis tissue labels directly.",
                "Generate a UMAP or PCA view with tissue annotations for the expanded-clonotype subset.",
                "Render or save visible figures in the notebook session.",
                "Print a pathway-oriented interpretation tied to the observed DE genes.",
            ],
            analysis_plan=[
                "Visualize transcriptional differences using UMAP or PCA with tissue annotations.",
                "Run expanded-clonotype differential expression between primary and metastatic tissues.",
                "Summarize top differentially expressed genes with a barplot and heatmap.",
                "Interpret the dominant functional programs or pathways represented by the top DE genes.",
            ],
            first_step_code=code,
            code_description=(
                "Generates a tissue-annotated embedding plus the required DE barplot and heatmap for "
                "expanded clonotypes, then prints a curated pathway interpretation for the top genes."
            ),
            summary=current_analysis.summary,
        )

    def interpret_results(
        self,
        notebook: nbf.NotebookNode,
        current_analysis,
        past_analyses: str,
        research_state_summary: str,
        step_validation_summary: str,
    ) -> str:
        last_code_cell = self._get_last_code_cell(notebook)
        if last_code_cell is None:
            return "No code cell was available to interpret."

        text_output = self._collect_text_output(last_code_cell)
        images = self._collect_image_outputs(last_code_cell)
        prompt = self._read_prompt("interp_results.txt").format(
            hypothesis=current_analysis.hypothesis,
            analysis_type=current_analysis.analysis_type,
            priority_question=current_analysis.priority_question,
            evidence_goal=current_analysis.evidence_goal,
            decision_rationale=current_analysis.decision_rationale,
            validation_checks="\n".join(f"- {item}" for item in current_analysis.validation_checks),
            analysis_plan="\n".join(f"- {step}" for step in current_analysis.analysis_plan),
            code=current_analysis.first_step_code,
            code_description=current_analysis.code_description,
            text_output=text_output or "No text output.",
            rna_summary=self.rna_summary,
            tcr_summary=self.tcr_summary,
            joint_summary=self.joint_summary,
            validation_summary=self.validation_summary,
            research_state=research_state_summary or "No research ledger entries yet.",
            step_validation_summary=step_validation_summary or "No step validation notes.",
            context_summary=self.context_summary,
            past_analyses=past_analyses or "No previous analyses.",
        )

        if self.use_VLM and images and self.vision_client is not None:
            try:
                content = [{"type": "text", "text": prompt}]
                for image in images:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image}"},
                        }
                    )
                response = self.vision_client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You interpret scRNA + scTCR notebook outputs and recommend the next research step.",
                        },
                        {"role": "user", "content": content},
                    ],
                )
                return response.choices[0].message.content or "No interpretation returned."
            except Exception as exc:
                self.logger.warning(f"Vision interpretation failed; falling back to text-only mode. Error: {exc}")

        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You interpret integrated scRNA + scTCR notebook outputs as a careful research scientist.",
                    },
                    {"role": "user", "content": prompt + f"\n\nNumber of figures produced: {len(images)}"},
                ],
            )
            return response.choices[0].message.content or "No interpretation returned."
        except Exception as exc:
            self.logger.warning(f"Text interpretation failed; using deterministic fallback. Error: {exc}")
            fallback_lines = [
                "### Deterministic Interpretation",
                "",
                f"- Step description: {current_analysis.code_description}",
                f"- Text output length: {len(text_output)} characters.",
                f"- Figures produced: {len(images)}.",
            ]
            if text_output:
                fallback_lines.extend(
                    [
                        "- Key text output excerpt:",
                        truncate_text(text_output, 1200),
                    ]
                )
            fallback_lines.extend(
                [
                    "- This fallback summary was generated locally because the model interpretation request failed.",
                    "- Review the notebook outputs directly before making strong biological claims.",
                ]
            )
            return "\n".join(fallback_lines)

    def fix_code(self, code: str, error_message: str, notebook: nbf.NotebookNode) -> str:
        documentation = ""
        if self.use_documentation:
            try:
                documentation = get_documentation(code)
            except Exception as exc:
                documentation = f"<documentation lookup failed: {exc}>"

        notebook_summary = truncate_text(summarize_notebook_cells(notebook.cells), 8000)
        documentation = truncate_text(documentation or "No documentation available.", 6000)
        trimmed_code = truncate_text(code, 6000)
        trimmed_error = truncate_text(error_message, 3000)
        prompt = self._read_prompt("fix_code.txt").format(
            current_code=trimmed_code,
            error_message=trimmed_error,
            notebook_summary=notebook_summary,
            documentation=documentation,
            available_packages=self.coding_guidelines,
        )
        response = litellm.completion(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You fix Python notebook code. Return only executable Python."},
                {"role": "user", "content": prompt},
            ],
        )
        return strip_code_fences(response.choices[0].message.content or code)

    def execute_idea(
        self,
        analysis,
        past_analyses: str,
        research_ledger: ResearchLedger,
        analysis_idx: int = 0,
        seeded: bool = False,
    ) -> tuple[str, ResearchLedger]:
        notebook_path = self.output_dir / f"{self.analysis_name}_analysis_{analysis_idx + 1}.ipynb"
        backup_path = self._backup_existing_notebook(notebook_path)
        if backup_path is not None:
            self.logger.info(f"Backed up existing notebook to {backup_path}")
        notebook = self.create_initial_notebook(analysis, research_ledger)

        plan_markdown = (
            "## Analysis Plan\n"
            f"Priority question: {analysis.priority_question}\n\n"
            f"Evidence goal: {analysis.evidence_goal}\n\n"
            f"Decision rationale: {analysis.decision_rationale}\n\n"
            "Validation checks:\n"
            + "\n".join(f"- {item}" for item in analysis.validation_checks)
            + "\n\nRemaining steps:\n"
            + "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(analysis.analysis_plan))
        )
        notebook.cells.append(new_markdown_cell(plan_markdown))
        approved_plan_items = list(analysis.analysis_plan)
        notebook.cells.append(
            new_markdown_cell(
                "## Approved Plan Contract\n\n"
                "The following plan items were approved by the user and must be completed before the run is considered complete:\n"
                + "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(approved_plan_items))
            )
        )

        self.start_persistent_kernel()
        last_interpretation = ""
        step_validation_summary = "No step validation notes yet."
        total_step_budget = max(self.max_iterations, len(approved_plan_items) + 2, 3)
        try:
            initial_code_cells = self._get_code_cells(notebook)
            if len(initial_code_cells) < 2:
                raise RuntimeError("Initial notebook must contain setup and baseline code cells.")
            ok, error = self._run_code_cell(initial_code_cells[0])
            if not ok:
                raise RuntimeError(f"Notebook setup failed: {error}")
            ok, error = self._run_code_cell(initial_code_cells[1])
            if not ok:
                raise RuntimeError(f"Baseline analysis setup failed: {error}")

            current_analysis = analysis

            for step_idx in range(total_step_budget):
                pending_plan_items, completed_plan_items = self._pending_plan_items(approved_plan_items, notebook)
                step_header = (
                    f"## Step {step_idx + 1} Summary\n\n"
                    f"{current_analysis.code_description}\n\n"
                    f"Priority question: {current_analysis.priority_question}\n\n"
                    f"Evidence goal: {current_analysis.evidence_goal}\n\n"
                    "Validation checks:\n" + "\n".join(f"- {item}" for item in current_analysis.validation_checks)
                )
                notebook.cells.append(new_markdown_cell(step_header))
                notebook.cells.append(new_code_cell(strip_code_fences(current_analysis.first_step_code)))

                self.update_code_memory(notebook.cells)
                success, error_message = self.run_last_code_cell(notebook)

                if not success:
                    self.logger.warning(
                        f"Analysis {analysis_idx + 1}, step {step_idx + 1} failed with error: {error_message}"
                    )
                    fixed = False
                    for fix_idx in range(self.max_fix_attempts):
                        repaired_code = self.fix_code(
                            current_analysis.first_step_code,
                            error_message or "Unknown error",
                            notebook,
                        )
                        notebook.cells[-1].source = repaired_code
                        current_analysis.first_step_code = repaired_code
                        success, error_message = self.run_last_code_cell(notebook)
                        if success:
                            fixed = True
                            self.logger.info(
                                f"Analysis {analysis_idx + 1}, step {step_idx + 1} fixed on attempt {fix_idx + 1}."
                            )
                            break
                    if not fixed:
                        failure_note = (
                            f"Step {step_idx + 1} failed after {self.max_fix_attempts} attempts. "
                            "The next step should change direction and avoid repeating the same error."
                        )
                        last_interpretation = failure_note

                last_code_cell = self._get_last_code_cell(notebook)
                text_output = self._collect_text_output(last_code_cell) if last_code_cell is not None else ""
                image_outputs = self._collect_image_outputs(last_code_cell) if last_code_cell is not None else []
                step_validation = self.step_validator.inspect_step_output(
                    current_analysis,
                    text_output=text_output,
                    image_count=len(image_outputs),
                    error_message=None if success else error_message,
                )
                pending_plan_items, completed_plan_items = self._pending_plan_items(approved_plan_items, notebook)
                enforcement_summary = (
                    "Approved-plan enforcement\n"
                    "Completed plan items:\n"
                    + ("\n".join(f"- {item}" for item in completed_plan_items) if completed_plan_items else "- none yet")
                    + "\n\nPending plan items:\n"
                    + ("\n".join(f"- {item}" for item in pending_plan_items) if pending_plan_items else "- none")
                )
                step_validation_summary = step_validation.to_prompt_text() + "\n\n" + enforcement_summary

                if success:
                    last_interpretation = self.interpret_results(
                        notebook,
                        current_analysis,
                        past_analyses,
                        research_ledger.to_prompt_text(),
                        step_validation_summary,
                    )

                notebook.cells.append(
                    new_markdown_cell(
                        f"## Step {step_idx + 1} Validation\n\n{step_validation.to_markdown()}"
                    )
                )
                notebook.cells.append(new_markdown_cell(f"## Plan Enforcement Snapshot\n\n{enforcement_summary}"))
                notebook.cells.append(
                    new_markdown_cell(
                        f"## Step {step_idx + 1} Interpretation\n\n{last_interpretation}"
                    )
                )

                step_update = self.hypothesis_generator.summarize_step_research(
                    current_analysis=current_analysis,
                    notebook_cells=notebook.cells,
                    text_output=text_output,
                    research_state_summary=research_ledger.to_prompt_text(),
                    step_validation_summary=step_validation_summary,
                )
                research_ledger.add_entry(step_update)
                notebook.cells.append(
                    new_markdown_cell(
                        "## Evidence Ledger Update\n\n"
                        f"Step title: {step_update.step_title}\n\n"
                        f"Status: {step_update.evidence_status}\n\n"
                        f"Claim: {step_update.claim}\n\n"
                        "Supporting evidence:\n"
                        + "\n".join(f"- {item}" for item in step_update.supporting_evidence)
                        + "\n\nCaveats:\n"
                        + "\n".join(f"- {item}" for item in step_update.caveats)
                        + "\n\nNext priority question:\n"
                        + step_update.next_priority_question
                        + "\n\nRecommended direction:\n"
                        + step_update.recommended_direction
                    )
                )
                notebook.cells.append(new_markdown_cell(f"## Research Ledger Snapshot\n\n{research_ledger.to_markdown()}"))

                self._save_notebook(notebook, notebook_path)
                if not pending_plan_items:
                    break

                steps_left = total_step_budget - step_idx - 1
                if steps_left <= 0 and not pending_plan_items:
                    break
                if steps_left <= 0 and pending_plan_items:
                    last_interpretation = (
                        (last_interpretation + "\n\n") if last_interpretation else ""
                    ) + "The run stopped with pending approved plan items still unfinished."
                    break

                fallback_analysis = self._build_plan_completion_fallback(
                    current_analysis,
                    pending_plan_items,
                    completed_plan_items,
                )
                if fallback_analysis is not None:
                    current_analysis = fallback_analysis
                else:
                    current_analysis = self.hypothesis_generator.generate_next_step(
                        current_analysis=current_analysis,
                        past_analyses=past_analyses,
                        notebook_cells=notebook.cells,
                        num_steps_left=steps_left,
                        research_state_summary=research_ledger.to_prompt_text(),
                        step_validation_summary=step_validation_summary,
                        approved_plan_items=approved_plan_items,
                        pending_plan_items=pending_plan_items,
                    )
                if pending_plan_items and not self._analysis_advances_pending_items(current_analysis, pending_plan_items):
                    fallback_analysis = self._build_plan_completion_fallback(
                        current_analysis,
                        pending_plan_items,
                        completed_plan_items,
                    )
                    if fallback_analysis is not None:
                        current_analysis = fallback_analysis
                    else:
                        strengthened_summary = (
                            step_validation_summary
                            + "\n\nThe proposed next step still does not clearly advance any pending approved plan item. "
                              "The next step must explicitly target one of the pending plan items."
                        )
                        current_analysis = self.hypothesis_generator.generate_next_step(
                            current_analysis=current_analysis,
                            past_analyses=past_analyses,
                            notebook_cells=notebook.cells,
                            num_steps_left=steps_left,
                            research_state_summary=research_ledger.to_prompt_text(),
                            step_validation_summary=strengthened_summary,
                            approved_plan_items=approved_plan_items,
                            pending_plan_items=pending_plan_items,
                        )
                        if pending_plan_items and not self._analysis_advances_pending_items(current_analysis, pending_plan_items):
                            fallback_analysis = self._build_plan_completion_fallback(
                                current_analysis,
                                pending_plan_items,
                                completed_plan_items,
                            )
                            if fallback_analysis is not None:
                                current_analysis = fallback_analysis

            final_pending_items, final_completed_items = self._pending_plan_items(approved_plan_items, notebook)
            notebook.cells.append(
                new_markdown_cell(
                    "## Final Summary\n\n"
                    f"{analysis.summary}\n\n"
                    f"{last_interpretation}\n\n"
                    "Approved plan completion:\n"
                    + ("Completed items:\n" + "\n".join(f"- {item}" for item in final_completed_items) if final_completed_items else "Completed items:\n- none")
                    + "\n\n"
                    + ("Pending items:\n" + "\n".join(f"- {item}" for item in final_pending_items) if final_pending_items else "Pending items:\n- none")
                    + "\n\n"
                    "## Final Research Ledger\n\n"
                    f"{research_ledger.to_markdown()}"
                )
            )
            self._save_notebook(notebook, notebook_path)
        finally:
            self.stop_persistent_kernel()

        summary_text = (
            f"Analysis {analysis_idx + 1}\n"
            f"Hypothesis: {analysis.hypothesis}\n"
            f"Type: {analysis.analysis_type}\n"
            f"Priority question: {analysis.priority_question}\n"
            f"Notebook: {notebook_path}\n"
            f"Final interpretation: {truncate_text(last_interpretation or analysis.summary, 800)}\n"
            f"Research ledger:\n{truncate_text(research_ledger.to_prompt_text(), 1200)}"
        )
        return (past_analyses + "\n\n" + summary_text).strip(), research_ledger

