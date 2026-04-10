# RetroFlowDPS Auto Research Program

## 理论优先原则（新增）

本项目实验以 `paper.tex` 中的理论链条为主线，不做“无假设参数乱扫”。

核心链条：

`fast conditional pass -> warm initialization -> lower conditional energy/free-energy -> better conditional init distribution -> reverse-flow inheritance -> better final metrics`

因此，任何 `sd3_sampler.py` 修改都必须能回答两个问题：

1. 该修改影响链条中的哪个环节？
2. 为什么预计会带来 `FID↓ / PSNR↑ / SSIM↑`？

---

## 目标

在固定评测协议下，持续优化 `RetroFlowDPS` 采样行为，目标为：

1. `FID` 降低（主目标，越低越好）
2. `PSNR` 升高（越高越好）
3. `SSIM` 升高（越高越好）

> 优先级：`FID` > `PSNR` > `SSIM`。

---

## 允许与禁止修改

### 允许修改

- `sd3_sampler.py`
  - 重点是 `RetroFlowDPS` 采样相关参数与策略，尤其：
    - `cfg_scale: float = 1.0`
    - `batch_size: int = 1`
    - `step_size: float = 15.0`
  - 允许在该文件内做与采样质量直接相关的最小必要改动。
- `paper.tex`（受控小改，新增）
  - 仅允许基于实验结果做“小幅理论-实证对齐”修改：
    - 补充更准确的假设边界、失败模式、适用条件
    - 微调方法描述与实验观察的一致性
    - 增加/修正消融结论文字
  - 不允许把未验证结论写成确定性结论。

### 禁止修改

- 不改数据集内容与标注。
- 不引入新依赖。
- 不改评测口径（PSNR/SSIM/FID 的计算方式保持现状）。
- 不做与采样质量无关的大规模重构。
- 不允许先改论文结论、后找实验“对齐”；必须先有可复现实验依据。

---

## 固定实验协议（必须一致）

在 `flow-matching` 根目录执行，固定命令模板：

```bash
torchrun --nproc_per_node=2 batch_solve.py \
  --use_ddp \
  --dataset configs/DIV2K_train.yml \
  --num_samples 4 \
  --method flowdps \
  --batch_size_per_gpu 2 \
  --clean_workdir > run.log 2>&1
```

说明：

- 使用相同 `num_samples`、相同数据配置做对比。
- 当前 `batch_solve.py` 对 `solver.sample()` 不再显式传 `cfg_scale/step_size`，因此实验主要通过 `sd3_sampler.py` 默认值与内部策略生效。
- 若改动涉及理论新假设，必须追加一组对照实验验证（至少 baseline + 新方案）。

---

## 实验初始化

1. 生成 run tag（建议 `apr10` / `apr10a` 这类）。
2. 从当前基线分支创建实验分支：

```bash
git checkout -b autoresearch/<tag>
```

3. 初始化 `results.tsv`（仅表头，tab 分隔）：

```tsv
commit	psnr	ssim	fid	status	description
```

4. 首次必须跑 baseline（不改代码），记录首行结果。

---

## 实验循环（自动化执行）

循环步骤：

1. 基于当前最佳提交，先写“理论假设一句话”（记录在 commit message 或实验备注）。
2. 修改 `sd3_sampler.py`（一次一个想法，改动小且可解释）。
   - 示例映射：
     - `cfg_scale`：影响条件势能下降强度（对应 Proposition 1 侧）
     - `step_size`：影响 early-stage 稳定性与继承误差（对应 Proposition 3 侧）
     - `batch_size`：仅在不改变评测口径前提下用于稳定统计，不作为主理论改动
3. 提交：

```bash
git add sd3_sampler.py
git commit -m "exp: <short description>"
```

4. 跑实验：

```bash
torchrun --nproc_per_node=2 batch_solve.py --use_ddp --dataset configs/DIV2K_train.yml --num_samples 4 --method flowdps --batch_size_per_gpu 2 --clean_workdir > run.log 2>&1
```

5. 读取指标：

```bash
grep -E "^  PSNR:|^  SSIM:|^  FID :" run.log
```

6. 记录到 `results.tsv`（**不要 commit 这个文件**），并写明“理论假设是否被支持”。
7. 若结果更优则保留提交；否则回退：

```bash
git reset --hard <上一个最佳commit>
```

---

## Keep / Discard 判定规则

设旧结果为 `(FID_old, PSNR_old, SSIM_old)`，新结果为 `(FID_new, PSNR_new, SSIM_new)`。

### Keep（满足其一）

- `FID_new < FID_old` 且 `PSNR_new >= PSNR_old - 0.02` 且 `SSIM_new >= SSIM_old - 0.002`
- 或 `FID_new` 持平（±0.1）但 `PSNR`、`SSIM` 同时提升
- 且理论解释与观测一致（至少不矛盾）

### Discard

- `FID` 变差明显（`+0.5` 以上），且没有显著画质补偿
- 指标基本持平但代码复杂度明显上升
- 指标变好但与理论方向长期冲突且无法复现

---

## 论文联动更新规则（新增）

当且仅当满足以下条件，才允许修改 `paper.tex`：

1. 同一结论在至少 2 次独立实验中复现（或在多个任务设置下一致）。
2. 有明确对照（baseline vs 新方案）。
3. 修改为“保守陈述”，不夸大因果。

建议修改位置：

- Method 部分：补充参数策略与理论映射。
- Ablation 部分：新增“何时有效/何时失效”的边界条件。
- Failure cases：写入已观测的不稳定区间（如过大 step 或过强 cfg）。

不建议修改：

- 未实证支持的 theorem 结论本体。
- 依赖额外训练或外部假设的新理论分支（除非后续专门实验覆盖）。

---

## 崩溃与超时处理

- 若 10 分钟仍未结束，判定超时，记为 `crash`。
- 若运行报错：
  - 明显小问题（拼写、shape、dtype）可快速修一次再跑。
  - 若方向本身不稳定，直接标 `crash` 并换下一个想法。

`results.tsv` 约定：

- 崩溃行填：`psnr=0.0000, ssim=0.000000, fid=9999.0000, status=crash`

---

## 建议优先实验方向（从简到繁）

1. `cfg_scale` 网格：`[0.8, 1.0, 1.2, 1.5, 2.0]`
2. `step_size` 网格：`[10, 12, 15, 18, 20]`
3. `cfg_scale` 与 `step_size` 联动（前期保守、后期增强，验证 early-stage 稳定性假设）
4. 空提示（null prompt）与条件提示混合权重微调
5. 对 batch 内样本采用一致/自适应参数的比较

每个方向都要写明“理论预期”：

- 若 `cfg_scale` 过低：条件约束不足，FID 可能恶化。
- 若 `cfg_scale` 过高：早期不稳定，PSNR/SSIM 可能下降。
- 若 `step_size` 过大：继承误差放大，结果波动加剧。
- 若 `step_size` 过小：收敛慢，有限 NFE 下可能欠优化。

---

## 日志记录示例

```tsv
commit	psnr	ssim	fid	status	description
a1b2c3d	22.7698	0.542654	162.1378	keep	baseline
b2c3d4e	22.9102	0.548120	158.9043	keep	cfg_scale 1.0->1.2
c3d4e5f	22.6001	0.531004	166.8821	discard	step_size 15->22
```

---

## 执行准则

- 始终保持单变量优先，避免一次改太多导致归因困难。
- 同等收益下，优先选择更简单实现。
- 每次只和“当前最佳提交”比较，不和历史差结果比较。
- 未经明确要求，不修改 `sd3_sampler.py` 之外的文件。
- 若修改 `paper.tex`，必须在 `results.tsv` 的 description 中标注对应证据实验编号。
