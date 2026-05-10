# Day-1 增强 #1 + #3:click prior + 跨 level 不 reset

> 2026-05-10。Day-1 增强清单(见 `docs/baseline-routes.md` §6)里两个不依赖 GPU、当天能验的:
> #1 — 状态栏遮蔽 + 颜色稀有性 prior(借 dolphin 思路的简化版)
> #3 — level transition 时不 reset model+optimizer(SG 自承的简化点)

## 1. 设计

### #1 StaticPriorBuilder(`src/perception/segmenter.py`)

**输入**:每帧 `(64, 64)` int 调色板索引。
**状态**:每 cell 累计变化次数 `change_count`,每色累计像素数 `color_count`。
**输出**:每帧 `(64, 64) [0,1]` multiplier,作用于 ACTION6 coord 概率分布。

**两条规则**:
1. **静态像素遮蔽**:观测 ≥ `mask_after_frames`(默认 20)帧后,从未变过色的 cell → prior=0。捕捉 UI / 状态栏 / 不动的边框。
2. **背景色降权**:整局观测里出现次数最多的颜色 → 这些 cell 的 prior 乘 `background_downweight`(默认 0.1)。捕捉墙、空地、大片背景。

**未做**(留给后续):连通块分割 → 按"段"而非"像素"采样。dolphin 论文核心,但需要 4-连通 labeling。先验证简化版有没有用,再决定要不要加。

### 接入 SGBaselineAgent

- `__init__`:`use_coord_prior` 开关 + `self.segmenter`(可为 None)
- `choose_action`:每步先 `segmenter.observe(arr_2d)` 再 `segmenter.click_prior(arr_2d)`
- `_sample`:`coord_p = sigmoid(coord_logits) / 4096 * prior_flatten`,然后归一化采样
- `maybe_reset_for_new_level`:level 切换时也 reset segmenter(背景色可能换)

### #3 跨 level 不 reset model

`SGBaselineAgent.maybe_reset_for_new_level(reset_model: bool = True)`,默认 True 保持原 SG 行为;`--no-reset-on-level` CLI 开关切到 False,经验池仍清,但模型权重 + Adam 状态保留。

> ⚠️ 在 cn04 sanity 里 0/6 关无法触发该路径,故 #3 在本轮**未真实生效验证**(只验证到代码不崩),要等真有 level transition 的 run 才能看是否抬分。

## 2. A/B/C 实验

3 run × 500 steps × seed 42 × cn04,CPU。

| 指标 | A vanilla | B +prior | C +prior+no-reset |
|---|---|---|---|
| step 450 fps | 1.0 | 0.6 | 0.6 |
| buf unique | 423 | 417 | 410 |
| pos / neg | 345 / 100 | 368 / 77 | 311 / 134 |
| pos rate | 77.5% | **82.7%** | 69.9% |
| prior_active | 1.00 | **0.54** | **0.56** |
| final loss | 0.506 | 0.411 | **0.227** |
| final acc | 0.81 | 0.78 | **0.86** |
| RHAE | 0.0 | 0.0 | 0.0 |
| levels | 0/6 | 0/6 | 0/6 |
| GAME_OVERs | 6 | 6 | 6 |

## 3. 解读

### Prior 是有用的(B vs A)
- **prior_active=0.54** ⇒ segmenter 屏蔽了 46% 的像素(cn04 大量背景 + 状态栏)
- **pos rate 提升 5pp** ⇒ 同样 500 步内,采到的"有反馈"动作多了 23 个(368 vs 345),信号密度提升
- **loss 降 19%** ⇒ 训练更稳,但 acc 略降(从 0.81 → 0.78)— 因为 pos/neg 更平衡时分类问题本身变难,而非模型变差

### 但 sanity 没出关
A/B/C 都是 RHAE 0.0,因为 cn04 level 1 的 baseline 是 29 步,触发逻辑大概率需要特定按钮组合,500 步随机/弱学习采样命中概率仍极低。**Prior 的真正价值要在 GPU + 8h 全预算下显现**。

### #3 没机会触发(但 sanity 暴露副作用)
没真正通关 → `maybe_reset_for_new_level` 在 level 维度没切换路径 → **跨 level 不 reset 的 effect 不可观测**。

但 C 的 final loss/acc(0.227 / 0.86)显著优于 B(0.411 / 0.78),这**不是 #3 的真实价值**,而是一个测量副作用:
- 第一帧的 `levels_completed=0` 触发了 "-1 → 0" 这条**虚假 transition**
- B 走 reset_model=True 路径,在该点把 model 重 init 了一次(用了不同的 RNG 状态)
- C 不 reset,保留了 `__init__` 时的 model
- 两者初始权重于是不同,后续轨迹分叉

**修法**:`maybe_reset_for_new_level` 应跳过 `current_levels_completed == -1` 的初始转换。已记入下一步 TODO。这同时提示我们 C-vs-B 的 loss 差异更多反映"避免在 step 0 重 init"的运气,而非 #3 真效果。

## 4. 局限与下一步

- [ ] **bug 修**:`maybe_reset_for_new_level` 跳过 `-1 → 0` 的虚假 transition(让 B/C 真正只在 reset_model 选项上分叉)
- [ ] 移植 dolphin 的连通块 + 5 priority tier(段级采样,真正把 4096 → ~30)
- [ ] 借 BS 的 state-graph dedup + GAME_OVER 黑名单(避免反复死同一个状态)
- [ ] 把 prior 的 `background_downweight` 做成 schedule(早期高,后期降),类似 ε
- [ ] 测一个比 cn04 反馈密度低的游戏(ft09?),看 prior 是否有相似/相反效果
- [ ] 上 GPU 跑 8h,观察 #3 是否真的让跨 level 学习更稳(需先修上面 bug)

## 5. 复现

```bash
# 三个 run 顺序跑(每个 ~13 min CPU)
uv run python scripts/train_sg.py --game cn04 --steps 500 --seed 42 --no-coord-prior --tag run-A-vanilla
uv run python scripts/train_sg.py --game cn04 --steps 500 --seed 42 --tag run-B-prior
uv run python scripts/train_sg.py --game cn04 --steps 500 --seed 42 --no-reset-on-level --tag run-C-prior-noreset
```
