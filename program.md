# RetroFlowDPS Auto Research Program

## 理论优先原则（新增）

本项目实验以 `paper.tex` 中的理论链条为主线，不做“无假设参数乱扫”。

核心链条：

`fast conditional pass -> warm initialization -> lower conditional energy/free-energy -> better conditional init distribution -> reverse-flow inheritance -> better final metrics`

因此，任何 `sd3_sampler.py` 修改都必须能回答两个问题：

1. 该修改影响链条中的哪个环节？
2. 为什么预计会带来 `FID↓ / PSNR↑ / SSIM↑`？

并且默认优先级为：

`sample 算法结构改进` > `数据一致性注入策略改进` > `小参数微调`

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
  - 重点是 `sample()` 算法行为本身（尤其 `flowdps` / `retroflowdps`）：
    - 两阶段采样结构（bootstrap / refine）
    - 回溯点策略（retro index / trigger 条件）
    - 重采样区间与噪声重注入公式
    - data consistency 的注入时机/频次/混合方式
    - 条件分支（null/cond）在时间轴上的调度
  - 允许在该文件内做与采样质量直接相关的最小必要改动。
  - `cfg_scale` / `step_size` 仅作为算法改动后的二阶微调，不作为主实验方向。
  - **硬约束**：`data_consistency()` 视为固定模块，不允许修改其实现。
  - **主优化位点**：仅在 `sample()` 内调整回溯策略（如 `retro_idx`、触发时机、回溯后重采样区间与流程）。
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
- 不允许修改 `data_consistency()` 的损失定义、梯度更新与迭代次数。

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
- 当前评测主战场是 `sd3_sampler.py::sample()` 的更新规则，不是默认超参数网格搜索。
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
     - 回溯触发与重采样设计：影响 warm initialization 质量（Proposition 1/2）
     - data consistency 注入调度：影响继承误差与稳定性（Proposition 3）
     - 噪声重注入公式/权重：影响 early-stage 偏差传播（Proposition 3）
     - `cfg_scale/step_size`：仅用于验证算法改动是否需要轻微配套校准
   - 本阶段新增限制：不改 `data_consistency()`，只改 `sample()` 中的回溯策略。
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

1. **[进行中] 流形平均噪声锚点（manifold-averaged noise anchor）**  
   在 first pass 每步记录 `z1t = z + (1-σ)·v`，回溯时用 `z1_avg = mean(z1_hist)` 代替单步反推的 `z1y` 重初始化，降低回溯起点方差。  
   - 理论依据：每步 `z1t` 是对相同源噪声 `z1 ~ N(0,I)` 的有偏估计；沿轨迹平均消减方差，使回溯起点更稳定（类比 FlowEdit source inversion，但无需参考图像）。  
   - 预期效果：PSNR/SSIM 提升（一致性更好），FID 降低（分布更紧）。  
   - 代码位点：`retroflowdps.sample()` 第一阶段 `z1_hist` 累积 + 回溯触发处。

2. 调整 `sample()` 的两阶段结构（是否回溯、回溯到何时、是否分段重采样）
3. 调整回溯触发时机（固定末步触发 vs 条件触发）
4. 调整回溯后重采样区间（从中段/后段重启）
5. 调整回溯后的噪声恢复路径（不改 data consistency 前提下）
6. 在回溯结构稳定后，再做 `cfg_scale/step_size` 小范围校准

每个方向都要写明"理论预期"：

- 若 `z1_avg` 效果好：说明单步 `z1y` 估计方差是主要误差来源，回溯起点质量可显著改善。
- 若 `z1_avg` 无改善：说明回溯起点不是瓶颈，需转向回溯点位置或重采样区间调整。
- 若回溯过早：warm start 信息不足，改进有限。
- 若回溯过晚：误差已累积，二次采样收益下降。
- 若一致性注入过强过早：细节损失、PSNR/SSIM 下降风险。
- 若重噪声过强：方差升高导致 FID 波动。
- 小参数（`cfg_scale/step_size`）只用于稳定算法，不替代算法创新。

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
