# CENSOR 本地缩小复现结果

这些文件根据当前项目中最新的六组完整实验自动生成。它们与论文图表采用相同的比较思路，但不是论文完整规模的数值复现。

## 输出文件

- `table1_local.png`：缩小版表 1。只统计 `epoch_0`，每个单元格为目标图像 0、1000、2000 的均值。
- `table1_local.csv`：表 1 的均值和样本标准差。
- `table1_local.xlsx`：带原始数据和公式汇总的 Excel 版本。
- `figure2_local.png`：无防御时，IG 与 GGL 在 epoch 0–4 的重建变化。
- `figure6_local.png`：目标图像 0 在 epoch 0 下，三种防御对 IG/GGL 的影响。
- `metrics_by_sample.csv`：六组实验、五个 epoch、三个目标图像的逐样本指标。

## 与论文的差别

- 论文表 1 包含 3 个数据集、5 种攻击、更多防御；这里仅有 FFHQ64、IG、GGL 和 3 种防御。
- 论文每个攻击做 4 次尝试并在 10 张图像上汇报平均值；这里使用当前保存的单次结果和 3 张图像。
- 表中 LPIPS 使用 VGG 版本，与论文的指标定义一致。
- 因此这些图表适合检查代码流程和观察趋势，不能直接宣称完整复现了论文数值。

## 本次选中的运行目录

- IG / No Defense: `results/geiping_ffhq/geiping_ffhq_none_num_images_1_defense_None_Sep.22_21.24.55`
- IG / Noise: `results/geiping_ffhq/geiping_ffhq_noise_num_images_1_defense_noise_Sep.22_23.22.58`
- IG / CENSOR: `results/geiping_ffhq/geiping_ffhq_censor_num_images_1_defense_orthogonal_Sep.22_17.18.28`
- GGL / No Defense: `results/ggl_ffhq/ggl_stylegan2_ffhq_none_num_images_1_defense_None_Sep.23_11.14.58`
- GGL / Noise: `results/ggl_ffhq/ggl_stylegan2_ffhq_noise_num_images_1_defense_noise_Sep.23_12.31.19`
- GGL / CENSOR: `results/ggl_ffhq/ggl_stylegan2_ffhq_censor_num_images_1_defense_orthogonal_Sep.23_14.51.31`

## 重新生成

在项目根目录和 `censor` 环境中执行：

```bash
python generate_reproduction_figures.py
```

脚本会自动选择每种配置最新的一组完整结果。可用 `--target-id` 更换图 2/图 6 展示的图像。
