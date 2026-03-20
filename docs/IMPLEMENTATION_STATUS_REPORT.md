# UCE 项目当前实现能力与实验结果报告

更新时间：2026-03-20
作者：Cursor Agent（本轮执行记录）

---

## 1. 已实现能力总览

当前项目已经从“脚手架状态”推进到“可运行基线 + 可跑 UQ 指标 + 可做参数矩阵实验”的阶段，核心能力如下。

### 1.1 数据获取与整理能力（DRIVE）

已实现统一数据准备脚本：`scripts/data/download_drive.py`，支持多级 fallback：

1. 官方/候选 URL 直链下载
2. 本地 zip 指定输入（`--training_zip` / `--test_zip`）
3. Kaggle CLI 下载（`--kaggle_dataset`）
4. KaggleHub Python 下载（`--kagglehub_dataset`）
5. 已解压 DRIVE 目录直接复用（即使 zip 不存在）

并实现自动标准化输出到：

- `data/processed/drive/train/images`
- `data/processed/drive/train/mask`
- `data/processed/drive/test/images`
- `data/processed/drive/test/mask`

其他处理能力：

- mask 二值化输出（0/255）
- 图像-标签文件名对齐
- 对某些镜像缺失 `test/1st_manual` 的情况自动回退到 `test/mask`

### 1.2 数据划分能力

已实现并可执行：`scripts/data/preprocess.py`

- 根据随机种子和 `val_ratio` 生成确定性 split 文件：
  - `data/splits/drive_train.txt`
  - `data/splits/drive_val.txt`
  - `data/splits/drive_test.txt`

当前实测：

- train=16
- val=4
- test=20

### 1.3 训练能力（Baseline UNet）

已打通并增强 `scripts/train/train_baseline.py`：

- 支持 DRIVE 真数据 + split file
- BCE + Dice 联合损失
- AdamW
- CosineAnnealingLR
- AMP（CUDA 可用时）
- best checkpoint / last checkpoint 保存
- 训练结束后自动导出完整 test 集预测
- test 推理默认加载 `best.pt`（而非 last）

产物路径：

- `outputs/*/checkpoints/best.pt`
- `outputs/*/checkpoints/last.pt`
- `outputs/*/predictions/*.png`

### 1.4 分割评估能力

已可执行 `scripts/eval/eval_seg.py`：

- 指标：Dice / IoU
- 兼容预测与 GT 尺寸不一致场景（中心裁剪对齐后计算）

### 1.5 UQ 推理能力

`scripts/infer/infer_uq.py` 已由随机张量版升级为真实 test dataloader 版：

- 支持方法：
  - deterministic
  - mc_dropout
  - deep_ensemble
  - tta
- 输入：真实 `DriveDataset(split="test")`
- 输出：`npz` 包含
  - `mean_prob`
  - `variance`
  - `entropy`
  - `names`（样本文件名）

### 1.6 正式 UQ 评估能力

`scripts/eval/eval_uq.py` 已实现正式指标：

- NLL
- Brier Score
- ECE
- Risk-Coverage / AURC
- Risk@100% coverage

并支持：

- 按 `names` 与 GT 对齐
- 预测尺寸与 GT 尺寸不一致时裁剪对齐

---

## 2. 使用的模型与训练设置

### 2.1 模型

主干模型：`UNet2D`

基础配置：`configs/models/unet2d.yaml`

- `in_channels: 3`
- `out_channels: 1`
- `init_features: 32`
- `dropout: 0.1`

矩阵中还使用：

- `configs/models/unet2d_matrix_b.yaml`：`init_features=48, dropout=0.1`
- `configs/models/unet2d_matrix_c.yaml`：`init_features=32, dropout=0.2`

### 2.2 训练策略

- 优化器：AdamW
- 学习率调度：CosineAnnealingLR
- 损失函数：
  \[
  \mathcal{L} = w_{bce}\cdot BCEWithLogits + w_{dice}\cdot DiceLoss
  \]
- 混合精度：AMP（可用时）
- 数据输入尺寸：`512x512`（中心裁剪）

---

## 3. 实验结果

> 说明：以下为当前已完成的真实运行结果（DRIVE 测试集 20 张，MC Dropout 评估）。

### 3.1 早期 baseline（全量 50 epoch）

- Dice: **0.2410**
- IoU: **0.1374**

对应 UQ（后续修正后同链路可运行）：

- NLL: 1.026279
- Brier: 0.401796
- ECE: 0.513040
- AURC: 0.560258
- Risk@100: 0.725725

### 3.2 参数实验矩阵（A/B/C）

#### Matrix A

配置：
- 模型：`unet2d.yaml`（32, dropout 0.1）
- 训练：`matrix_a.yaml`（lr=1e-4, bce=0.3, dice=0.7, epoch=80）

结果：
- Dice: **0.2673**
- IoU: **0.1545**
- NLL: **1.078143**
- Brier: **0.420988**
- ECE: **0.531434**
- AURC: **0.467805**
- Risk@100: **0.713721**

#### Matrix B

配置：
- 模型：`unet2d_matrix_b.yaml`（48, dropout 0.1）
- 训练：`matrix_b.yaml`（lr=1e-4, bce=0.2, dice=0.8, epoch=80）

结果：
- Dice: **0.2877**（当前最好）
- IoU: **0.1683**（当前最好）
- NLL: **1.145680**
- Brier: **0.442811**
- ECE: **0.538219**
- AURC: **0.558921**
- Risk@100: **0.703680**

#### Matrix C

配置：
- 模型：`unet2d_matrix_c.yaml`（32, dropout 0.2）
- 训练：`matrix_c.yaml`（lr=8e-5, bce=0.3, dice=0.7, epoch=80）

结果：
- Dice: **0.2760**
- IoU: **0.1604**
- NLL: **0.789459**（当前最好）
- Brier: **0.298470**（当前最好）
- ECE: **0.425240**（当前最好）
- AURC: **0.461994**（当前最好）
- Risk@100: **0.680659**（当前最好）

### 3.3 结果解读

- **分割精度优先**：Matrix B 更优（Dice/IoU 最好）。
- **UQ 可靠性优先**：Matrix C 明显更优（NLL/Brier/ECE/AURC 最优）。
- 当前存在明显 trade-off：提升分割主指标时，校准/不确定性指标可能变差。

---

## 4. 新增与修改的代码/文件清单

> 以下为本轮推进过程中新增或重点修改的核心文件。

### 4.1 新增文件

1. `scripts/data/download_drive.py`
   - DRIVE 数据下载、解压、标准化总入口
2. `configs/training/matrix_a.yaml`
3. `configs/training/matrix_b.yaml`
4. `configs/training/matrix_c.yaml`
5. `configs/models/unet2d_matrix_b.yaml`
6. `configs/models/unet2d_matrix_c.yaml`
7. `configs/experiments/exp_matrix_a.yaml`
8. `configs/experiments/exp_matrix_b.yaml`
9. `configs/experiments/exp_matrix_c.yaml`
10. `docs/IMPLEMENTATION_STATUS_REPORT.md`（本文档）

### 4.2 关键修改文件

1. `src/uce/data/dataset.py`
   - 支持 `split_file`
   - 扩展图像后缀
2. `scripts/train/train_baseline.py`
   - 本地导入稳定化
   - test dataloader 接入
   - full epoch 行为修复
   - best checkpoint 推理导出
3. `scripts/eval/eval_seg.py`
   - 尺寸不一致裁剪对齐
4. `scripts/infer/infer_uq.py`
   - 从随机输入升级到真实 test dataloader
   - 输出 `names`
5. `scripts/eval/eval_uq.py`
   - 实现正式 UQ 指标（NLL/Brier/ECE/AURC）
6. `docs/NEXT_STEP_PLAN.md`
   - 多轮状态与命令更新

---

## 5. 数据与产物现状

### 5.1 数据

- 原始数据：`data/raw/drive/DRIVE/...`
- 处理后数据：`data/processed/drive/...`
- split 文件：`data/splits/drive_*.txt`

### 5.2 实验产物

- 基线与矩阵输出目录：
  - `outputs/`
  - `outputs/matrix_a/`
  - `outputs/matrix_b/`
  - `outputs/matrix_c/`
- 典型产物：
  - `checkpoints/best.pt`, `checkpoints/last.pt`
  - `predictions/*.png`
  - `uq_maps/mc_dropout_stats.npz`

---

## 6. 当前能力边界与下一步建议

### 6.1 当前边界

- 训练与评估已可复现，但 Dice 仍偏低（小数据、增强不足、网络/阈值未充分调优）。
- UQ 指标已可计算，但尚未纳入统一 CSV 自动汇总导出。

### 6.2 建议下一步

1. 引入更强数据增强（几何 + 光照）
2. 增加阈值调优（val 集最优阈值）
3. 增加温度标定（进一步降 ECE）
4. 将 Seg + UQ 指标统一写入 `reports/tables/exp01_metrics.csv`
5. 增加实验自动 runner（批量配置执行 + 汇总）

---

## 7. 常用复现实验命令

```bash
# 数据准备
python scripts/data/download_drive.py --kagglehub_dataset andrewmvd/drive-digital-retinal-images-for-vessel-extraction
python scripts/data/preprocess.py --root data/processed/drive --seed 42 --val_ratio 0.2

# 训练（示例：matrix_c）
python scripts/train/train_baseline.py --config configs/experiments/exp_matrix_c.yaml --smoke_steps 0

# 分割评估
python scripts/eval/eval_seg.py --pred_dir outputs/matrix_c/predictions --gt_dir data/processed/drive/test/mask

# UQ 推理 + 评估
python scripts/infer/infer_uq.py \
  --config configs/experiments/exp_matrix_c.yaml \
  --method mc_dropout \
  --ckpt outputs/matrix_c/checkpoints/best.pt \
  --output outputs/matrix_c/uq_maps/mc_dropout_stats.npz \
  --num_samples 10 --batch_size 1

python scripts/eval/eval_uq.py \
  --uq_npz outputs/matrix_c/uq_maps/mc_dropout_stats.npz \
  --gt_dir data/processed/drive/test/mask --num_bins 15
```
