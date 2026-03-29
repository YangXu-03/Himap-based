#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


METHODS = [
    {
        "name": "attn",
        "title": "Attn",
        "dir": "attn_heatmaps",
        "pattern": "heatmap_{category}_top{pct}pct.png",
    },
    {
        "name": "text2img",
        "title": "Text2Img",
        "dir": "attn_heatmaps_text2img",
        "pattern": "heatmap_prompt2img_{category}_top{pct}pct.png",
    },
    {
        "name": "promptweight",
        "title": "PromptWeight",
        "dir": "attn_heatmaps_promptweighted",
        "pattern": "heatmap_promptweighted_{category}_top{pct}pct.png",
    },
]

TOP_PCTS = [1, 10, 20, 50]


def find_categories(root: Path) -> list[str]:
    base_dir = root / METHODS[0]["dir"]
    categories: set[str] = set()
    for image_path in base_dir.glob("heatmap_*_top*pct.png"):
        stem = image_path.stem
        if not stem.startswith("heatmap_"):
            continue
        if "_top" not in stem:
            continue
        category = stem[len("heatmap_") : stem.rfind("_top")]
        if category:
            categories.add(category)
    return sorted(categories)


def all_images_exist(root: Path, category: str) -> bool:
    for method in METHODS:
        method_dir = root / method["dir"]
        for pct in TOP_PCTS:
            img_name = method["pattern"].format(category=category, pct=pct)
            if not (method_dir / img_name).exists():
                return False
    return True


def draw_grid(root: Path, out_dir: Path, category: str) -> Path:
    fig, axes = plt.subplots(len(TOP_PCTS), len(METHODS), figsize=(12, 14))
    fig.suptitle(f"Category: {category}", fontsize=16, y=0.995)

    for row_idx, pct in enumerate(TOP_PCTS):
        for col_idx, method in enumerate(METHODS):
            ax = axes[row_idx, col_idx]
            img_path = root / method["dir"] / method["pattern"].format(category=category, pct=pct)
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(method["title"], fontsize=12, pad=8)

            if col_idx == 0:
                ax.set_ylabel(f"top {pct}%", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out_path = out_dir / f"heatmap_{category}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def draw_per_top(root: Path, out_dir: Path, category: str, pct: int) -> Path:
    fig, axes = plt.subplots(1, len(METHODS), figsize=(12, 4.4))
    fig.suptitle(f"Category: {category} | top {pct}%", fontsize=15, y=1.01)

    for col_idx, method in enumerate(METHODS):
        ax = axes[col_idx]
        img_path = root / method["dir"] / method["pattern"].format(category=category, pct=pct)
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(method["title"], fontsize=12, pad=8)

    plt.tight_layout()
    out_path = out_dir / f"heatmap_{category}_top{pct}pct.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compose heatmaps from attn, text2img, and promptweight into one large figure "
            "for each category, with rows as top-k percentages and columns as methods."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root path (default: script directory).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "combined_heatmaps",
        help="Output directory (default: ./combined_heatmaps).",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Optional categories to process. If omitted, process all available categories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = args.categories if args.categories else find_categories(root)
    if not categories:
        raise SystemExit("No categories found in attn_heatmaps.")

    processed = 0
    skipped: list[str] = []

    for category in categories:
        if not all_images_exist(root, category):
            skipped.append(category)
            continue

        grid_out = draw_grid(root, out_dir, category)
        print(f"Saved grid: {grid_out}")

        for pct in TOP_PCTS:
            per_top_out = draw_per_top(root, out_dir, category, pct)
            print(f"Saved per-top: {per_top_out}")

        processed += 1

    print(f"Processed categories: {processed}")
    if skipped:
        print(f"Skipped categories (missing files): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
