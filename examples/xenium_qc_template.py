"""Template for Xenium QC.

This file is intentionally non-executing documentation. Replace paths and run
manually only when working in an approved analysis environment.
"""

from spatialcore import qc
from spatialcore.io import read_h5ad


def xenium_qc_template(h5ad_path):
    adata = read_h5ad(h5ad_path)
    adata = qc.flag_mito_genes(adata, prefix="mt-", inplace=False)
    qc_df, summary_df, metadata = qc.compute_qc_metrics(adata)
    fig, axes = qc.plot_qc_histograms(adata, show=True, save_path=None)
    return {"adata": adata, "qc": qc_df, "summary": summary_df, "metadata": metadata, "figure": fig}
