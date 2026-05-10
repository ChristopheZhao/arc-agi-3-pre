# Baseline 1:SG-port + click prior(候选之一)

> 这是 `docs/baseline-routes.md` 中 **A 路线**(CNN frame-change predictor)的当前实现快照。
> 不是定型架构;只要 GPU benchmark 出分后 fallback 信号触发,我们会切到 B(state-graph)或 C(训练-free 探索),这份文档**保留作为对照**。
>
> 状态:本地 sanity 验证通过,Kaggle P100 上首次见到 cn04 lvl 1 通关。

## 0. 一句话定位

**在线学习的 CNN bandit**:每步预测"下一动作会不会改变 64×64 屏幕",按预测概率采样动作。无外部 reward,frame-change 自创密集信号。

## 1. 数据流(每一步发生什么)

```
ARC-AGI-3 服务端
    │ env.reset() / env.step(action, data)
    ▼
FrameData {
  frame: list[N_subframes][64][64] palette idx,
  state, levels_completed, win_levels,
  available_actions: list[int]   ← 关键 mask 源
}
    │
    │ scripts/{train_sg, benchmark}.py 主循环
    ▼
┌──────── SGBaselineAgent.choose_action(frame) ────────────┐
│                                                            │
│  ① _frame_2d → 取 frame[-1] 最后子帧 (64,64)              │
│                                                            │
│  ② StaticPriorBuilder                                      │
│     - observe(arr): 累 change_count + color_count          │
│     - click_prior(arr) → (64,64) [0,1] mult                │
│       静态像素(20+ 帧没变)→ 0;modal 背景色 → ×0.1        │
│                                                            │
│  ③ one-hot tensor (16,64,64) → ActionModel                 │
│     conv1-4: 16→32→64→128→256 (3×3 same)                   │
│     ┌─ action_head: maxpool4 → 512 → 5                     │
│     └─ coord_head: 4 conv → 1×64×64 → flatten 4096         │
│     output: cat([5, 4096]) = 4101 logits                   │
│                                                            │
│  ④ _sample(logits, available_actions, prior)               │
│     - mask: 不在 available_actions 的位置 → -inf           │
│     - coord prob ×= prior.flatten()                         │
│     - 拼成 (5+4096),归一化,multinomial 采样              │
│     - idx<5 → ACTION1-5;idx≥5 → ACTION6(y, x)             │
│                                                            │
└────────────────────────────────────────────────────────────┘
    │ 返回 (GameAction, data, unified_action_idx)
    ▼
env.step → new FrameData
    │
    ▼
┌──────── agent.observe(prev_state, prev_action, new) ─────┐
│  - reward = 1 if frame_changed else 0                     │
│  - hash = MD5(prev_frame_bool + str(action_idx))          │
│  - 若新 hash:经验池 deque 追加 {state, action_idx, reward}│
└────────────────────────────────────────────────────────────┘
    │
    ▼ (每 5 个 env step)
┌──────── agent.train_step() ──────────────────────────────┐
│  - 经验池采 batch 64                                       │
│  - forward → 在 action_idx 位置 gather logit               │
│  - loss = BCE(selected_logit, reward)                      │
│           - 1e-4 · action_entropy                          │
│           - 1e-5 · coord_entropy                           │
│  - Adam lr=1e-4 step                                       │
└────────────────────────────────────────────────────────────┘
```

## 2. 三个核心设计选择(为什么这么做)

### A. 训"frame change",不训 reward
ARC-AGI-3 几乎无外部 reward — 唯一信号 `levels_completed +1` 平均 100+ 步出现一次。frame change **密集、自带、免费**,作为 reward shaping 代理。冠军 SG 用此,我们继承。

### B. BCE 只对采到的那个 logit 做
我们只观测到**实际选中**的 (state, action) 改没改帧;其他 4100 个 (state, action') 没监督源 ⇒ 不施加损失。off-policy bandit 标准做法。

### C. ACTION6 用卷积输出 4096 logits,不分 (x_logits, y_logits)
独立 (x, y) 假设结果可分离 — **错的**(点 (0,0) 与 (63,0) 后果可完全不同)。卷积 head 保留 2D 邻接性。SG 原作明示 flatten 形态不稳。

## 3. 模块职责一览

| 模块 | 输入 | 输出 | 关键决策 |
|---|---|---|---|
| `src/agents/sg_baseline.py::ActionModel` | (B,16,64,64) one-hot | (B,4101) logits | 5+4096 拼 head;coord 用卷积 |
| `src/agents/sg_baseline.py::SGBaselineAgent` | FrameData stream | GameAction stream | 真 level transition 时 reset 模型(可关);MD5 dedup;train every 5 step |
| `src/perception/segmenter.py::StaticPriorBuilder` | (64,64) palette idx 流 | (64,64) [0,1] mask | 20 帧后静态遮蔽;modal color ×0.1 |
| `scripts/train_sg.py` | 1 game | logs + scorecard | 开发用,A/B flag 全开 |
| `scripts/benchmark.py` | N games | runs/<tag>/* | 时间预算 + 可恢复 + per-game fresh agent |
| `notebooks/benchmark_p100.ipynb` | 上面两脚本 | runs.zip | Kaggle P100 入口 |

## 4. 已验证 / 已观察短板

### ✅ 验证有效
- click prior 在 cn04 屏蔽 46% 像素,pos rate +5pp,loss -19%(见 `docs/day1-enhancements.md` A/B/C 数据)
- P100 ~38 fps(vs CPU 1.2)
- cn04 lvl 1 通过(本地 CPU sanity 没做到)

### 🔴 已观察硬伤
- **lvl-2+ 退化**:cn04 lvl 1 通后 11000 步 lvl 2 没动。几乎所有动作都改帧 ⇒ frame-change 信号失去判别力,只有特定动作序列才能通关
- **per-game 模型**:每开新游戏从零开始,无跨任务先验
- **只看最后一帧**:N_subframes 中的动画过渡信息丢弃
- **无动作历史**:纯 state-conditional,无时序上下文
- **prior 不学习**:规则式静态 mask,而非神经网络判断

## 5. 与候选 baseline 2/3 的关系

参考 `docs/baseline-routes.md` 路线对比:

| 路线 | 当前实现 | 何时切换到这路 |
|---|---|---|
| **A. SG-port + click prior(本文)** | ✅ baseline-1,Kaggle P100 跑 benchmark 中 | 默认起点 |
| B. state-graph + ResNet 价值网(BS 思路) | 未实现 | A 在 lvl-2+ 全卡 / 模式坍塌 → 切 B 用段级动作 + 价值网 |
| C. 训练-free 段级探索(dolphin 思路) | 未实现 | A 在多游戏 frame-change AUC 维持 0.5(没学到信号)→ 切 C 跳过 NN |

**baseline-1 不是终点**。它的价值是:
1. 验证 Kaggle 端到端流程通(GPU 选型 / clone / install / API / scorecard)
2. 给出**真实 RHAE 数字**作为后续路线判断的对照
3. 暴露具体硬伤(上面 🔴),把"Day-2 最该攻什么"从直觉变成数据驱动

## 6. Day-2 改进候选(等 benchmark 出全数据后排序)

按 `docs/baseline-routes.md` §6 列举,**优先级取决于 25 demo 跑出的分布**:

1. **段级动作空间**(dolphin 思路):4-连通块 + tier 优先级,把 4096 点击压成 ~30 段。**不是 fine-tune,是动作空间本身要变**。如果 cn04 类型(背景大、按钮少)是 25 game 主流 ⇒ 杠杆最大
2. **state-graph + GAME_OVER 黑名单**(BS 思路):显式记录死路径,推理时硬避开。如果多游戏 reset 频繁 ⇒ 救命
3. **多帧 attention**(更激进):N_subframes 作为时间维度,小 transformer 捕"动作进行中"信息。如果 lvl-2+ 卡 = 时序信息缺失,这是根本解法
4. **跨游戏迁移**:共享 backbone,新游戏只 fine-tune head。**12.58% 冠军没做这个**,但应是更高上限的方向
5. **真 schedule prior 强度**:`background_downweight` 从 0.1 早期高 → 后期降,模拟 ε 退火

## 7. 依赖与部署双轨

| 环境 | 来源 | 关键差异 |
|---|---|---|
| 本地 WSL2 CPU 开发 | `pyproject.toml` + `uv.lock` | torch 2.11+cpu |
| Kaggle P100 benchmark | `requirements-kaggle.txt` | torch 2.5.1+cu121(sm_60 兼容,绕开 Kaggle 预装的 2.10+cu128) |
| Kaggle 真提交(未来) | 待做,需 Dataset 化 + 离线 wheel | 评测无网 |

## 8. 复现路径

本地 sanity:
```bash
uv sync
uv run python scripts/train_sg.py --game cn04 --steps 500 --seed 42
```

Kaggle P100 benchmark:
```bash
# 见 docs/kaggle-notebook-setup.md;upload notebooks/benchmark_p100.ipynb
```
