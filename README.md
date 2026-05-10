# arc-agi-3-pre

ARC Prize 2026 — ARC-AGI-3 赛道的预调研与 baseline 工作仓。

> 比赛主页:<https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3>
> 赛道说明:<https://arcprize.org/competitions/2026/arc-agi-3>
> SDK 文档:<https://docs.arcprize.org/>

## 比赛要点

- **格式**:交互式回合制游戏。无规则说明,agent 看 64×64 RGB(16 色)帧,输出动作。
- **动作空间**:`ACTION1–5`(无参)、`ACTION6(x, y)`(64×64 点击,4096 维)、`ACTION7`(撤销)。
- **评分**:RHAE — 相对人类(第二名)动作效率,跨游戏归一化。
- **奖金**:Grand $700K + Top Score $75K + Milestones $75K。
- **关键约束**:Kaggle 评测**无网络**;**全部代码必须开源**。
- **关键日期**:2026-03-25 开赛 / 2026-06-30 M1 / 2026-09-30 M2 / 2026-11-02 提交截止。
- **当前水位**:人类 100%;前沿 LLM <1%(Gemini 3.1 Pro 0.37%);预览赛冠军 12.58%(CNN+RL)。

## 仓库结构

```
.
├── pyproject.toml          # uv 管理的依赖
├── docs/                   # 技术报告、SDK notes
├── scripts/                # quickstart / dump_frames 等一次性脚本
├── src/                    # baseline agent 与公共模块(后续添加)
└── ref/                    # 外部参考仓(不入库,见 .gitignore)
    └── ARC-AGI-3-Agents/   # arcprize 官方 agent 样例
```

## 快速上手

```bash
# 装依赖(已锁在 uv.lock)
uv sync

# 设置 API key(从 https://three.arcprize.org 拿)
cp .env.example .env  # 然后编辑填入 ARC_API_KEY

# 跑 quickstart
uv run python scripts/quickstart.py
```

## 路线候选(Phase 4 决策)

| 路线 | 思路 | 预览赛战绩 | 工程量 |
|---|---|---|---|
| A. State-graph + 动作过滤 | 状态图+剪枝+价值排序 | Blind Squirrel 6.71% | 中 |
| B. CNN + RL 动作预测 | 学"哪些动作改帧" | StochasticGoose 12.58% 🥇 | 中-高 |
| C. World model (Dreamer/IRIS) | 学环境动力学+latent 规划 | — | 高 |
