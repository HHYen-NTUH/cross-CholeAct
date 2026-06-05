"""
Domain-gap analysis for cross-CholeAct technical validation.
=========================================================
Produces:
  1. umap_dataset_comparison.png        — combined UMAP coloured by dataset
  2. umap_perclass_domgap_legend.png    — 7 per-class UMAPs + summary panel (with annotations & legend)
  3. umap_perclass_domgap_nolabel.png   — same layout, clean version (no annotations / legend)


The input .npz files are produced by:

  python run_class_finetuning.py ... --eval --extract_eval_features

Expected arrays: features, labels. Optional metadata arrays such as video_ids,
predictions, logits, dataset_source, and split_name are ignored by this script.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = {
    0: "Dissecting",
    1: "Exposing",
    2: "Cutting",
    3: "Suctioning/Irrigating",
    5: "Coagulating",
    6: "Clipping/Unclipping",
    7: "Idle",
}


def get_args():
    parser = argparse.ArgumentParser(
        "Feature-space domain-gap analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--private_features", required=True, help="feature .npz for the private/cross-CholeAct split")
    parser.add_argument("--public_features", required=True, help="feature .npz for the regenerated Cholec80-Action split")
    parser.add_argument("--output_dir", default="./domain_gap_outputs", help="folder for UMAP figures")
    parser.add_argument("--private_name", default="cross-CholeAct", help="label shown for the private dataset")
    parser.add_argument("--public_name", default="Cholec80-Action", help="label shown for the public dataset")
    parser.add_argument("--n_neighbors", default=30, type=int, help="UMAP n_neighbors")
    parser.add_argument("--min_dist", default=0.1, type=float, help="UMAP min_dist")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    return parser.parse_args()


def load_feature_npz(path):
    data = np.load(path, allow_pickle=True)
    if "features" not in data or "labels" not in data:
        raise KeyError(f"{path} must contain 'features' and 'labels' arrays")
    return data["features"].astype(np.float32), data["labels"].astype(np.int64)


def rbf_kernel_cross(a, b, sigma2, batch=200):
    total = 0.0
    for i in range(0, len(a), batch):
        ab = a[i:i + batch]
        d2 = np.sum((ab[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        total += np.exp(-d2 / (2 * sigma2)).sum()
    return total / (len(a) * len(b))


def rbf_kernel_self(a, sigma2, batch=200):
    n = len(a)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(0, n, batch):
        ab = a[i:i + batch]
        d2 = np.sum((ab[:, None, :] - a[None, :, :]) ** 2, axis=-1)
        k = np.exp(-d2 / (2 * sigma2))
        lo = i
        hi = min(i + batch, n)
        k[np.arange(hi - lo), np.arange(lo, hi)] = 0.0
        total += k.sum()
    return total / (n * (n - 1))


def mmd_rbf(x, y, n_sub=1000, seed=42):
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    if len(x) > n_sub:
        x = x[rng.choice(len(x), n_sub, replace=False)]
    if len(y) > n_sub:
        y = y[rng.choice(len(y), n_sub, replace=False)]
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    xy = np.concatenate([x[:200], y[:200]])
    sub_d2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1)
    positive = sub_d2[sub_d2 > 0]
    sigma2 = float(np.median(positive)) if len(positive) else 1.0
    return float(rbf_kernel_self(x, sigma2) + rbf_kernel_self(y, sigma2) - 2 * rbf_kernel_cross(x, y, sigma2))


def present_classes(labels_public, labels_private):
    labels = sorted(set(labels_public.tolist()) | set(labels_private.tolist()))
    return [label for label in labels if label in CLASS_NAMES]


def save_dataset_umap(path, emb_public, emb_private, args):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(emb_public[:, 0], emb_public[:, 1], s=4, alpha=0.4, c="royalblue", rasterized=True)
    ax.scatter(emb_private[:, 0], emb_private[:, 1], s=4, alpha=0.4, c="mediumseagreen", rasterized=True)
    ax.set_title(f"UMAP after PCA-50: {args.public_name} vs {args.private_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_aspect("equal", "datalim")
    ax.legend([args.public_name, args.private_name], framealpha=0.85)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved {os.path.basename(path)}")


def draw_perclass_figure(
    path,
    classes,
    emb_public,
    emb_private,
    labels_public,
    labels_private,
    perclass_mmd,
    perclass_cdist,
    global_mmd,
    dataset_cdist,
    args,
    with_legend,
):
    x_all = np.concatenate([emb_public[:, 0], emb_private[:, 0]])
    y_all = np.concatenate([emb_public[:, 1], emb_private[:, 1]])
    xlim = (x_all.min() - 0.5, x_all.max() + 0.5)
    ylim = (y_all.min() - 0.5, y_all.max() + 0.5)

    n_panels = len(classes) + 1
    n_cols = 4
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5.5 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="royalblue", markersize=6, label=args.public_name),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="mediumseagreen", markersize=6, label=args.private_name),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="royalblue", markersize=10, label=f"{args.public_name} centroid"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="mediumseagreen", markersize=10, label=f"{args.private_name} centroid"),
        Line2D([0], [0], color="black", lw=1.2, ls="--", alpha=0.7, label="Centroid link"),
    ]

    for i, cls in enumerate(classes):
        ax = axes[i]
        mask_public = labels_public == cls
        mask_private = labels_private == cls
        ax.scatter(emb_public[:, 0], emb_public[:, 1], s=2, alpha=0.07, c="lightgrey", rasterized=True)
        ax.scatter(emb_private[:, 0], emb_private[:, 1], s=2, alpha=0.07, c="lightgrey", rasterized=True)
        ax.scatter(emb_public[mask_public, 0], emb_public[mask_public, 1], s=5, alpha=0.55, c="royalblue", rasterized=True)
        ax.scatter(emb_private[mask_private, 0], emb_private[mask_private, 1], s=5, alpha=0.55, c="mediumseagreen", rasterized=True)

        cx_public = emb_public[mask_public, 0].mean()
        cy_public = emb_public[mask_public, 1].mean()
        cx_private = emb_private[mask_private, 0].mean()
        cy_private = emb_private[mask_private, 1].mean()
        ax.scatter(cx_public, cy_public, s=200, c="royalblue", marker="*", edgecolors="white", linewidths=0.8, zorder=5)
        ax.scatter(cx_private, cy_private, s=200, c="mediumseagreen", marker="*", edgecolors="white", linewidths=0.8, zorder=5)
        ax.plot([cx_public, cx_private], [cy_public, cy_private], color="black", lw=1.2, ls="--", alpha=0.7, zorder=4)

        if with_legend:
            info = f"MMD = {perclass_mmd[cls]:.3f}\nDelta centroid = {perclass_cdist[cls]:.2f}"
            ax.text(0.03, 0.97, info, transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75, ec="grey"))
            ax.legend(handles=handles, fontsize=6.8, loc="lower right", framealpha=0.85)

        ax.set_title(f"{cls}: {CLASS_NAMES[cls]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")

    ax_summary = axes[len(classes)]
    ax_summary_r = ax_summary.twinx()
    mmd_vals = [perclass_mmd[c] for c in classes]
    cdist_vals = [perclass_cdist[c] for c in classes]
    x = np.arange(len(classes))
    width = 0.38
    ax_summary.bar(x - width / 2, mmd_vals, width=width, color="steelblue", alpha=0.8)
    ax_summary_r.bar(x + width / 2, cdist_vals, width=width, color="coral", alpha=0.8)
    ax_summary.axhline(global_mmd, color="steelblue", ls="--", lw=1.4, alpha=0.9)
    ax_summary_r.axhline(dataset_cdist, color="dimgrey", ls=":", lw=1.6, alpha=0.9)
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels([CLASS_NAMES[c][:10] for c in classes], rotation=35, ha="right", fontsize=8)
    ax_summary.set_ylabel("MMD", color="steelblue", fontsize=9)
    ax_summary_r.set_ylabel("Centroid distance", color="coral", fontsize=9)
    ax_summary.set_title("Domain Gap Summary", fontsize=11, fontweight="bold")

    if with_legend:
        summary_handles = [
            Line2D([0], [0], color="steelblue", lw=7, alpha=0.8, label="Per-class MMD"),
            Line2D([0], [0], color="steelblue", lw=1.4, ls="--", label=f"Global MMD = {global_mmd:.4f}"),
            Line2D([0], [0], color="coral", lw=7, alpha=0.8, label="Per-class centroid distance"),
            Line2D([0], [0], color="dimgrey", lw=1.6, ls=":", label=f"Dataset centroid distance = {dataset_cdist:.2f}"),
        ]
        ax_summary.legend(handles=summary_handles, fontsize=7, loc="upper right", framealpha=0.85)

    for ax in axes[n_panels:]:
        ax.axis("off")

    plt.suptitle(
        f"Per-class UMAP | {args.public_name} vs {args.private_name} | "
        f"Global MMD = {global_mmd:.4f} | Dataset centroid distance = {dataset_cdist:.2f}",
        fontsize=11,
        y=1.005,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {os.path.basename(path)}")


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading feature dumps...")
    feat_private, labels_private = load_feature_npz(args.private_features)
    feat_public, labels_public = load_feature_npz(args.public_features)
    classes = present_classes(labels_public, labels_private)
    print(f"  {args.public_name}:  {feat_public.shape[0]} samples")
    print(f"  {args.private_name}: {feat_private.shape[0]} samples")
    print(f"  classes: {classes}")

    print("\nRunning PCA-50 on combined features...")
    all_feats = np.concatenate([feat_public, feat_private], axis=0)
    all_scaled = StandardScaler().fit_transform(all_feats)
    n_components = min(50, all_scaled.shape[0], all_scaled.shape[1])
    all_pca = PCA(n_components=n_components, random_state=args.seed).fit_transform(all_scaled)
    print("Running UMAP...")
    embedding = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
    ).fit_transform(all_pca)

    n_public = len(feat_public)
    emb_public = embedding[:n_public]
    emb_private = embedding[n_public:]

    print("\nComputing MMD and centroid distances...")
    global_mmd = mmd_rbf(feat_public, feat_private, n_sub=1000, seed=args.seed)
    print(f"  Global MMD: {global_mmd:.6f}")
    dataset_cdist = float(np.linalg.norm(feat_public.mean(0) - feat_private.mean(0)))
    print(f"  Dataset centroid distance: {dataset_cdist:.4f}")

    print(
        f"\n{'Class':<28} {('n_' + args.public_name):>10} "
        f"{('n_' + args.private_name):>12} {'MMD':>10} {'Centroid dist':>14}"
    )
    print("-" * 78)
    perclass_mmd = {}
    perclass_cdist = {}
    for cls in classes:
        x = feat_public[labels_public == cls]
        y = feat_private[labels_private == cls]
        perclass_mmd[cls] = mmd_rbf(x, y, n_sub=500, seed=args.seed)
        perclass_cdist[cls] = float(np.linalg.norm(x.mean(0) - y.mean(0))) if len(x) and len(y) else float("nan")
        print(
            f"{cls}: {CLASS_NAMES[cls]:<24} {len(x):>10} {len(y):>12} "
            f"{perclass_mmd[cls]:>10.6f} {perclass_cdist[cls]:>14.4f}"
        )

    print("\nPlotting Figure 1: dataset comparison UMAP...")
    save_dataset_umap(os.path.join(args.output_dir, "umap_dataset_comparison.tiff"), emb_public, emb_private, args)
    print("\nPlotting Figure 2 & 3: per-class UMAPs...")
    draw_perclass_figure(
        os.path.join(args.output_dir, "umap_perclass_domgap_legend.tiff"),
        classes,
        emb_public,
        emb_private,
        labels_public,
        labels_private,
        perclass_mmd,
        perclass_cdist,
        global_mmd,
        dataset_cdist,
        args,
        with_legend=True,
    )
    draw_perclass_figure(
        os.path.join(args.output_dir, "umap_perclass_domgap_nolabel.tiff"),
        classes,
        emb_public,
        emb_private,
        labels_public,
        labels_private,
        perclass_mmd,
        perclass_cdist,
        global_mmd,
        dataset_cdist,
        args,
        with_legend=False,
    )
    print(f"\nAll done. Outputs saved under {args.output_dir}")


if __name__ == "__main__":
    main()
