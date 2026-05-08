"""Annotation, label-transfer, and SpatialData table-color helpers."""

from __future__ import annotations

import colorsys
from typing import Any, Mapping, Sequence

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from .utils import optional_import, require_obsm


def transfer_labels(
    query: Any,
    reference: Any,
    label_key: str,
    use_rep: str = "X_pca",
    n_neighbors: int = 15,
    output_key: str | None = None,
):
    """Transfer labels from a reference AnnData object by KNN in a shared representation."""
    if label_key not in reference.obs:
        raise KeyError(f"`reference.obs['{label_key}']` was not found.")
    if use_rep == "spatial":
        X_ref = require_obsm(reference, "spatial")
        X_query = require_obsm(query, "spatial")
    else:
        if use_rep not in reference.obsm or use_rep not in query.obsm:
            raise KeyError(f"`{use_rep}` must exist in both `.obsm` mappings.")
        X_ref = np.asarray(reference.obsm[use_rep])
        X_query = np.asarray(query.obsm[use_rep])
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    clf.fit(X_ref, reference.obs[label_key].astype(str))
    out = query.copy()
    out.obs[output_key or f"{label_key}_transferred"] = clf.predict(X_query)
    if hasattr(clf, "predict_proba"):
        out.obs[f"{output_key or label_key}_transfer_score"] = clf.predict_proba(X_query).max(axis=1)
    return out


def run_celltypist(adata: Any, model: str | None = None, majority_voting: bool = True, **kwargs: Any):
    """Run CellTypist lazily and return its annotation result."""
    celltypist = optional_import("celltypist", "xenium")
    return celltypist.annotate(adata, model=model, majority_voting=majority_voting, **kwargs)


def prepare_scvi_anndata(
    adata: Any,
    layer: str = "counts",
    batch_key: str | None = None,
    labels_key: str | None = None,
    **kwargs: Any,
):
    """Call `scvi.model.SCVI.setup_anndata` with notebook-style defaults."""
    scvi = optional_import("scvi", "xenium")
    scvi.model.SCVI.setup_anndata(adata, layer=layer, batch_key=batch_key, labels_key=labels_key, **kwargs)
    return adata


def _shades(base_hex: str, n: int, start: float = 0.35, end: float = 0.8) -> list[str]:
    if n <= 0:
        return []
    if n == 1:
        return [mcolors.to_hex(base_hex)]
    h, lightness, saturation = colorsys.rgb_to_hls(*mcolors.to_rgb(base_hex))
    return [
        mcolors.to_hex(colorsys.hls_to_rgb(h, value, saturation))
        for value in np.linspace(start, end, n)
    ]


def standardize_celltype_colors(
    adata: Any,
    label_col: str,
    celltype_groups: Mapping[str, Sequence[str]] | None = None,
    group_base_colors: Mapping[str, str] | None = None,
    unassigned_color: str = "#bdbdbd",
) -> dict[str, str]:
    """Set Scanpy-compatible `{label_col}_colors` from grouped palettes."""
    if label_col not in adata.obs:
        raise KeyError(f"`adata.obs['{label_col}']` was not found.")
    adata.obs[label_col] = pd.Categorical(adata.obs[label_col].astype(str))
    cats = list(adata.obs[label_col].cat.categories)

    label_to_color: dict[str, str] = {}
    if celltype_groups and group_base_colors:
        missing = [group for group in celltype_groups if group not in group_base_colors]
        if missing:
            raise KeyError(f"Missing base colors for groups: {missing}")
        for group, labels in celltype_groups.items():
            for label, color in zip(labels, _shades(group_base_colors[group], len(labels))):
                label_to_color[str(label)] = color
    else:
        palette = list(mcolors.TABLEAU_COLORS.values())
        for idx, cat in enumerate(cats):
            label_to_color[cat] = mcolors.to_hex(palette[idx % len(palette)])

    colors = [label_to_color.get(cat, unassigned_color) for cat in cats]
    adata.uns[f"{label_col}_colors"] = colors
    return dict(zip(cats, colors))


def rebuild_cropped_table(
    sdata_small: Any,
    sdata_original: Any,
    labels_element: str = "cell_labels",
    table_name: str = "table",
    region_key: str | None = None,
    instance_key_source: str | None = None,
    instance_key_out: str = "label_ids",
):
    """Rebuild a cropped SpatialData table using label ids from a cropped labels element."""
    models = optional_import("spatialdata.models", "spatial")
    arr = sdata_small[labels_element]["scale0"]["image"]
    label_ids = np.unique(arr.data.compute() if hasattr(arr.data, "compute") else np.asarray(arr.data))
    label_ids = label_ids[label_ids != 0].astype(int)

    adata = sdata_original[table_name].copy()
    attrs = adata.uns.get("spatialdata_attrs", {})
    region_key = region_key or attrs.get("region_key", "region")
    instance_key_source = instance_key_source or attrs.get("instance_key")

    if instance_key_source and instance_key_source in adata.obs:
        vals = pd.to_numeric(adata.obs[instance_key_source], errors="coerce")
    else:
        vals = pd.Series(np.arange(1, adata.n_obs + 1), index=adata.obs_names)
    adata.obs[instance_key_out] = vals.astype("Int64")
    mask = adata.obs[instance_key_out].isin(label_ids)
    if region_key in adata.obs:
        mask &= adata.obs[region_key].astype(str).eq(labels_element)
    adata_small = adata[mask].copy()
    if adata_small.n_obs == 0:
        raise ValueError("No table rows matched the cropped label ids.")
    adata_small.obs[region_key] = pd.Categorical([labels_element] * adata_small.n_obs)
    return models.TableModel.parse(
        adata_small,
        region=labels_element,
        region_key=region_key,
        instance_key=instance_key_out,
        overwrite_metadata=True,
    )
