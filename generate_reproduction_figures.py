#!/usr/bin/env python3
"""Generate a compact local reproduction of CENSOR Table 1, Figure 2 and Figure 6.

The script reads the latest completed runs for:
  - IG (Geiping): no defense / noise / CENSOR
  - GGL:          no defense / noise / CENSOR

It intentionally uses only the Python standard library and Pillow, so it does
not require pandas or matplotlib. Run it from the repository root:

    python generate_reproduction_figures.py

Outputs are written to results/summary by default.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "summary"
EPOCHS = tuple(range(5))


@dataclass(frozen=True)
class RunSpec:
    attack: str
    method_dir: str
    defense: str
    parent: str
    glob_pattern: str


RUN_SPECS: Tuple[RunSpec, ...] = (
    RunSpec("IG", "geiping", "No Defense", "geiping_ffhq", "geiping_ffhq_none_num_images_1_defense_None_*"),
    RunSpec("IG", "geiping", "Noise", "geiping_ffhq", "geiping_ffhq_noise_num_images_1_defense_noise_*"),
    RunSpec("IG", "geiping", "CENSOR", "geiping_ffhq", "geiping_ffhq_censor_num_images_1_defense_orthogonal_*"),
    RunSpec("GGL", "ggl", "No Defense", "ggl_ffhq", "ggl_stylegan2_ffhq_none_num_images_1_defense_None_*"),
    RunSpec("GGL", "ggl", "Noise", "ggl_ffhq", "ggl_stylegan2_ffhq_noise_num_images_1_defense_noise_*"),
    RunSpec("GGL", "ggl", "CENSOR", "ggl_ffhq", "ggl_stylegan2_ffhq_censor_num_images_1_defense_orthogonal_*"),
)

DEFENSE_ORDER = ("No Defense", "Noise", "CENSOR")
ATTACK_ORDER = ("IG", "GGL")

COLORS = {
    "navy": "#17324D",
    "blue": "#2F6690",
    "light_blue": "#DCEAF5",
    "pale_blue": "#EEF5FA",
    "green": "#DDEEDB",
    "gray": "#66717E",
    "light_gray": "#E5E9ED",
    "very_light": "#F7F9FB",
    "text": "#17212B",
    "white": "#FFFFFF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: results/summary)",
    )
    parser.add_argument(
        "--target-id",
        type=int,
        default=0,
        help="Target image ID used in Figure 2 and Figure 6 (default: 0)",
    )
    parser.add_argument(
        "--figure6-epoch",
        type=int,
        default=0,
        choices=EPOCHS,
        help="Epoch used in Figure 6 (paper uses the initial round; default: 0)",
    )
    return parser.parse_args()


def _font_candidates(bold: bool) -> Iterable[Path]:
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    roots = (
        Path("C:/Windows/Fonts"),
        Path("/mnt/c/Windows/Fonts"),
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/msttcorefonts"),
    )
    for root in roots:
        for name in names:
            yield root / name


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for path in _font_candidates(bold):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = COLORS["text"],
) -> None:
    width, height = text_size(draw, text, font)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: Tuple[int, int],
    end: Tuple[int, int],
    fill: str,
    width: int = 3,
    dash: int = 12,
    gap: int = 9,
) -> None:
    x1, y1 = start
    x2, y2 = end
    if x1 == x2:
        y = y1
        while y < y2:
            draw.line((x1, y, x2, min(y + dash, y2)), fill=fill, width=width)
            y += dash + gap
    elif y1 == y2:
        x = x1
        while x < x2:
            draw.line((x, y1, min(x + dash, x2), y2), fill=fill, width=width)
            x += dash + gap
    else:
        raise ValueError("Only horizontal or vertical dashed lines are supported")


def is_complete_run(path: Path, spec: RunSpec) -> bool:
    return all(
        (path / f"epoch_{epoch}" / "table_Metrics.csv").is_file()
        and (path / f"epoch_{epoch}" / spec.method_dir).is_dir()
        for epoch in EPOCHS
    )


def find_latest_run(spec: RunSpec) -> Path:
    parent = REPO_ROOT / "results" / spec.parent
    candidates = [p for p in parent.glob(spec.glob_pattern) if p.is_dir() and is_complete_run(p, spec)]
    if not candidates:
        raise FileNotFoundError(
            f"No completed run found for {spec.attack} / {spec.defense}. "
            f"Expected under: {parent / spec.glob_pattern}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def metric_columns(spec: RunSpec) -> Mapping[str, str]:
    prefix = spec.method_dir
    return {
        "MSE": f"{prefix}_mse_i",
        "LPIPS (VGG)": f"{prefix}_lpips(vgg)",
        "LPIPS (Alex)": f"{prefix}_lpips(alex)",
        "PSNR": f"{prefix}_psnr",
        "SSIM": f"{prefix}_ssim",
    }


def read_metrics(run: Path, spec: RunSpec, epoch: int) -> List[Dict[str, float]]:
    csv_path = run / f"epoch_{epoch}" / "table_Metrics.csv"
    columns = metric_columns(spec)
    rows: List[Dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in ("target_id", *columns.values()) if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing columns {missing} in {csv_path}")
        for source in reader:
            rows.append(
                {
                    "target_id": int(float(source["target_id"])),
                    **{metric: float(source[column]) for metric, column in columns.items()},
                }
            )
    if not rows:
        raise ValueError(f"No metric rows found in {csv_path}")
    return rows


def locate_image(run: Path, spec: RunSpec, epoch: int, target_id: int, generated: bool) -> Path:
    filename = f"{target_id}_gen.png" if generated else f"{target_id}_gt.png"
    path = run / f"epoch_{epoch}"
    if generated:
        path = path / spec.method_dir
    path = path / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing image: {path}")
    return path


def load_square(path: Path, size: int = 256) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.width != image.height:
            side = min(image.width, image.height)
            left = (image.width - side) // 2
            top = (image.height - side) // 2
            image = image.crop((left, top, left + side, top + side))
        return image.resize((size, size), Image.Resampling.NEAREST)


def write_metrics_csv(
    output_dir: Path,
    selected_runs: Mapping[Tuple[str, str], Tuple[RunSpec, Path]],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for attack in ATTACK_ORDER:
        for defense in DEFENSE_ORDER:
            spec, run = selected_runs[(attack, defense)]
            for epoch in EPOCHS:
                for row in read_metrics(run, spec, epoch):
                    records.append(
                        {
                            "Dataset": "FFHQ64",
                            "Attack": attack,
                            "Defense": defense,
                            "Epoch": epoch,
                            "Target ID": row["target_id"],
                            "MSE": row["MSE"],
                            "LPIPS (VGG)": row["LPIPS (VGG)"],
                            "LPIPS (Alex)": row["LPIPS (Alex)"],
                            "PSNR": row["PSNR"],
                            "SSIM": row["SSIM"],
                            "Run directory": str(run.relative_to(REPO_ROOT)),
                        }
                    )

    path = output_dir / "metrics_by_sample.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    return records


def summarize_epoch_zero(records: Sequence[Mapping[str, object]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    summary: Dict[Tuple[str, str], Dict[str, float]] = {}
    for attack in ATTACK_ORDER:
        for defense in DEFENSE_ORDER:
            selected = [
                row
                for row in records
                if row["Attack"] == attack and row["Defense"] == defense and row["Epoch"] == 0
            ]
            if not selected:
                raise ValueError(f"No epoch_0 data for {attack} / {defense}")
            values: Dict[str, float] = {"N": float(len(selected))}
            for metric in ("MSE", "LPIPS (VGG)", "PSNR", "SSIM"):
                metric_values = [float(row[metric]) for row in selected]
                values[metric] = statistics.fmean(metric_values)
                values[f"{metric} SD"] = statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0
            summary[(attack, defense)] = values
    return summary


def write_table_csv(output_dir: Path, summary: Mapping[Tuple[str, str], Mapping[str, float]]) -> None:
    metrics = ("MSE", "LPIPS (VGG)", "PSNR", "SSIM")
    headers = ["Dataset", "Defense"]
    for attack in ATTACK_ORDER:
        for metric in metrics:
            headers.extend((f"{attack} {metric} mean", f"{attack} {metric} SD"))
    headers.extend(("Epoch", "Images per cell"))

    path = output_dir / "table1_local.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for defense in DEFENSE_ORDER:
            row: Dict[str, object] = {"Dataset": "FFHQ64", "Defense": defense, "Epoch": 0}
            for attack in ATTACK_ORDER:
                values = summary[(attack, defense)]
                row["Images per cell"] = int(values["N"])
                for metric in metrics:
                    row[f"{attack} {metric} mean"] = values[metric]
                    row[f"{attack} {metric} SD"] = values[f"{metric} SD"]
            writer.writerow(row)


def best_cells(summary: Mapping[Tuple[str, str], Mapping[str, float]]) -> set[Tuple[str, str, str]]:
    best: set[Tuple[str, str, str]] = set()
    directions = {"MSE": max, "LPIPS (VGG)": max, "PSNR": min, "SSIM": min}
    for attack in ATTACK_ORDER:
        for metric, chooser in directions.items():
            values = {defense: summary[(attack, defense)][metric] for defense in DEFENSE_ORDER}
            selected_value = chooser(values.values())
            for defense, value in values.items():
                if math.isclose(value, selected_value, rel_tol=1e-12, abs_tol=1e-12):
                    best.add((attack, defense, metric))
    return best


def make_table_png(
    output_dir: Path,
    summary: Mapping[Tuple[str, str], Mapping[str, float]],
) -> Path:
    path = output_dir / "table1_local.png"
    widths = [240] + [225] * 8
    row_heights = [105, 70, 82, 82, 82, 80]
    margin_x, margin_y, footer_h = 55, 45, 65
    canvas_w = sum(widths) + 2 * margin_x
    canvas_h = sum(row_heights) + 2 * margin_y + footer_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), COLORS["white"])
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(36, bold=True)
    header_font = get_font(24, bold=True)
    body_font = get_font(25)
    body_bold = get_font(25, bold=True)
    note_font = get_font(20)

    draw.text((margin_x, margin_y), "Local Table 1 — FFHQ64, epoch 0", font=title_font, fill=COLORS["navy"])
    y = margin_y + row_heights[0]

    draw.rectangle((margin_x, y, margin_x + widths[0], y + row_heights[1]), fill=COLORS["navy"])
    draw_centered(draw, (margin_x + widths[0] / 2, y + row_heights[1] / 2), "Defense", header_font, COLORS["white"])
    x = margin_x + widths[0]
    for attack in ATTACK_ORDER:
        group_w = sum(widths[1:5])
        draw.rectangle((x, y, x + group_w, y + row_heights[1]), fill=COLORS["blue"])
        draw_centered(draw, (x + group_w / 2, y + row_heights[1] / 2), attack, header_font, COLORS["white"])
        x += group_w

    y += row_heights[1]
    x = margin_x
    labels = ["", "MSE ↑", "LPIPS ↑", "PSNR ↓", "SSIM ↓", "MSE ↑", "LPIPS ↑", "PSNR ↓", "SSIM ↓"]
    for index, (label, width) in enumerate(zip(labels, widths)):
        fill = COLORS["light_blue"] if index else COLORS["pale_blue"]
        draw.rectangle((x, y, x + width, y + row_heights[2]), fill=fill, outline=COLORS["white"], width=2)
        draw_centered(draw, (x + width / 2, y + row_heights[2] / 2), label, header_font)
        x += width

    best = best_cells(summary)
    metrics = ("MSE", "LPIPS (VGG)", "PSNR", "SSIM")
    formats = {"MSE": ".4f", "LPIPS (VGG)": ".4f", "PSNR": ".3f", "SSIM": ".4f"}
    for row_index, defense in enumerate(DEFENSE_ORDER):
        y += row_heights[2 + row_index]
        row_h = row_heights[3 + row_index]
        x = margin_x
        row_fill = COLORS["white"] if row_index % 2 == 0 else COLORS["very_light"]
        values: List[Tuple[str, bool]] = [(defense, defense == "CENSOR")]
        for attack in ATTACK_ORDER:
            for metric in metrics:
                values.append((format(summary[(attack, defense)][metric], formats[metric]), (attack, defense, metric) in best))
        for cell_index, ((label, bold), width) in enumerate(zip(values, widths)):
            fill = COLORS["green"] if bold and cell_index > 0 else row_fill
            draw.rectangle((x, y, x + width, y + row_h), fill=fill)
            draw.line((x, y + row_h, x + width, y + row_h), fill=COLORS["light_gray"], width=2)
            draw_centered(draw, (x + width / 2, y + row_h / 2), label, body_bold if bold else body_font)
            x += width

    note_y = canvas_h - margin_y - 35
    note = "Mean over target IDs 0, 1000 and 2000. LPIPS uses VGG. Highlight = strongest privacy metric in this local comparison."
    draw.text((margin_x, note_y), note, font=note_font, fill=COLORS["gray"])
    canvas.save(path, dpi=(300, 300))
    return path


def make_figure2(
    output_dir: Path,
    selected_runs: Mapping[Tuple[str, str], Tuple[RunSpec, Path]],
    target_id: int,
) -> Path:
    path = output_dir / "figure2_local.png"
    image_size = 256
    left_label = 105
    gap = 24
    title_h = 120
    row_h = 340
    columns = 6
    margin = 45
    canvas_w = margin * 2 + left_label + columns * image_size + (columns - 1) * gap
    canvas_h = margin * 2 + title_h + 2 * row_h + 55
    canvas = Image.new("RGB", (canvas_w, canvas_h), COLORS["white"])
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(36, bold=True)
    label_font = get_font(25, bold=True)
    small_font = get_font(22)

    draw.text((margin, margin), "Local Figure 2 — No defense across the first five epochs", font=title_font, fill=COLORS["navy"])
    header_y = margin + title_h - 25
    start_x = margin + left_label
    for col, label in enumerate(("Original", "Epoch 0", "Epoch 1", "Epoch 2", "Epoch 3", "Epoch 4")):
        x = start_x + col * (image_size + gap) + image_size / 2
        draw_centered(draw, (x, header_y), label, label_font)

    separator_x = start_x + image_size + gap / 2
    draw_dashed_line(draw, (int(separator_x), int(header_y + 25)), (int(separator_x), canvas_h - margin - 38), COLORS["gray"], 3)

    for row_index, attack in enumerate(ATTACK_ORDER):
        spec, run = selected_runs[(attack, "No Defense")]
        y = margin + title_h + row_index * row_h
        draw_centered(draw, (margin + left_label / 2, y + image_size / 2), attack, label_font, COLORS["navy"])
        original = load_square(locate_image(run, spec, 0, target_id, generated=False), image_size)
        canvas.paste(original, (start_x, y))
        for epoch in EPOCHS:
            image = load_square(locate_image(run, spec, epoch, target_id, generated=True), image_size)
            x = start_x + (epoch + 1) * (image_size + gap)
            canvas.paste(image, (x, y))
        draw_centered(draw, (start_x + (columns * image_size + (columns - 1) * gap) / 2, y + image_size + 30), f"Target ID {target_id}", small_font, COLORS["gray"])

    draw.text((margin, canvas_h - margin - 20), "FFHQ64. IG and GGL only; the paper includes five attacks.", font=small_font, fill=COLORS["gray"])
    canvas.save(path, dpi=(300, 300))
    return path


def make_figure6(
    output_dir: Path,
    selected_runs: Mapping[Tuple[str, str], Tuple[RunSpec, Path]],
    target_id: int,
    epoch: int,
) -> Path:
    path = output_dir / "figure6_local.png"
    image_size = 300
    left_label = 190
    gap = 36
    title_h = 130
    row_h = 380
    margin = 55
    columns = 3
    canvas_w = margin * 2 + left_label + columns * image_size + (columns - 1) * gap
    canvas_h = margin * 2 + title_h + len(DEFENSE_ORDER) * row_h + 50
    canvas = Image.new("RGB", (canvas_w, canvas_h), COLORS["white"])
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(38, bold=True)
    label_font = get_font(27, bold=True)
    small_font = get_font(22)

    draw.text((margin, margin), f"Local Figure 6 — Defense comparison at epoch {epoch}", font=title_font, fill=COLORS["navy"])
    header_y = margin + title_h - 30
    start_x = margin + left_label
    for col, label in enumerate(("Original", "IG", "GGL")):
        x = start_x + col * (image_size + gap) + image_size / 2
        draw_centered(draw, (x, header_y), label, label_font)
    separator_x = start_x + image_size + gap / 2
    draw_dashed_line(draw, (int(separator_x), int(header_y + 28)), (int(separator_x), canvas_h - margin - 35), COLORS["gray"], 3)

    base_spec, base_run = selected_runs[("IG", "No Defense")]
    original = load_square(locate_image(base_run, base_spec, epoch, target_id, generated=False), image_size)
    for row_index, defense in enumerate(DEFENSE_ORDER):
        y = margin + title_h + row_index * row_h
        label_color = COLORS["blue"] if defense != "CENSOR" else COLORS["navy"]
        draw_centered(draw, (margin + left_label / 2, y + image_size / 2), defense, label_font, label_color)
        canvas.paste(original, (start_x, y))
        for attack_index, attack in enumerate(ATTACK_ORDER):
            spec, run = selected_runs[(attack, defense)]
            image = load_square(locate_image(run, spec, epoch, target_id, generated=True), image_size)
            x = start_x + (attack_index + 1) * (image_size + gap)
            canvas.paste(image, (x, y))

    draw.text((margin, canvas_h - margin - 20), f"FFHQ64, target ID {target_id}. Reduced comparison: 2 attacks × 3 defenses.", font=small_font, fill=COLORS["gray"])
    canvas.save(path, dpi=(300, 300))
    return path


def write_readme(
    output_dir: Path,
    selected_runs: Mapping[Tuple[str, str], Tuple[RunSpec, Path]],
    target_id: int,
    figure6_epoch: int,
) -> None:
    lines = [
        "# CENSOR 本地缩小复现结果",
        "",
        "这些文件根据当前项目中最新的六组完整实验自动生成。它们与论文图表采用相同的比较思路，但不是论文完整规模的数值复现。",
        "",
        "## 输出文件",
        "",
        "- `table1_local.png`：缩小版表 1。只统计 `epoch_0`，每个单元格为目标图像 0、1000、2000 的均值。",
        "- `table1_local.csv`：表 1 的均值和样本标准差。",
        "- `table1_local.xlsx`：带原始数据和公式汇总的 Excel 版本。",
        "- `figure2_local.png`：无防御时，IG 与 GGL 在 epoch 0–4 的重建变化。",
        f"- `figure6_local.png`：目标图像 {target_id} 在 epoch {figure6_epoch} 下，三种防御对 IG/GGL 的影响。",
        "- `metrics_by_sample.csv`：六组实验、五个 epoch、三个目标图像的逐样本指标。",
        "",
        "## 与论文的差别",
        "",
        "- 论文表 1 包含 3 个数据集、5 种攻击、更多防御；这里仅有 FFHQ64、IG、GGL 和 3 种防御。",
        "- 论文每个攻击做 4 次尝试并在 10 张图像上汇报平均值；这里使用当前保存的单次结果和 3 张图像。",
        "- 表中 LPIPS 使用 VGG 版本，与论文的指标定义一致。",
        "- 因此这些图表适合检查代码流程和观察趋势，不能直接宣称完整复现了论文数值。",
        "",
        "## 本次选中的运行目录",
        "",
    ]
    for attack in ATTACK_ORDER:
        for defense in DEFENSE_ORDER:
            _, run = selected_runs[(attack, defense)]
            lines.append(f"- {attack} / {defense}: `{run.relative_to(REPO_ROOT)}`")
    lines.extend(
        [
            "",
            "## 重新生成",
            "",
            "在项目根目录和 `censor` 环境中执行：",
            "",
            "```bash",
            "python generate_reproduction_figures.py",
            "```",
            "",
            "脚本会自动选择每种配置最新的一组完整结果。可用 `--target-id` 更换图 2/图 6 展示的图像。",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_runs: Dict[Tuple[str, str], Tuple[RunSpec, Path]] = {}
    for spec in RUN_SPECS:
        selected_runs[(spec.attack, spec.defense)] = (spec, find_latest_run(spec))

    records = write_metrics_csv(output_dir, selected_runs)
    summary = summarize_epoch_zero(records)
    write_table_csv(output_dir, summary)
    outputs = [
        make_table_png(output_dir, summary),
        make_figure2(output_dir, selected_runs, args.target_id),
        make_figure6(output_dir, selected_runs, args.target_id, args.figure6_epoch),
    ]
    write_readme(output_dir, selected_runs, args.target_id, args.figure6_epoch)

    print("Selected runs:")
    for attack in ATTACK_ORDER:
        for defense in DEFENSE_ORDER:
            _, run = selected_runs[(attack, defense)]
            print(f"  {attack:>3} / {defense:<10}: {run.relative_to(REPO_ROOT)}")
    print("Generated:")
    for output in outputs:
        print(f"  {output}")
    print(f"  {output_dir / 'table1_local.csv'}")
    print(f"  {output_dir / 'metrics_by_sample.csv'}")
    print(f"  {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
