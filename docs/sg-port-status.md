# SG Baseline 移植状态

> 2026-05-10 完成 path β。Path α(SG 仓 verbatim)被 WSL2 + /mnt/d 磁盘满 + 无 GPU 直通联手挡死,改为把 SG 核心搬进 `src/agents/sg_baseline.py`。

## 状态:✅ 端到端跑通,等 GPU 出真分

### 产物
- `src/agents/sg_baseline.py` — `ActionModel`(34.3M 参数)+ `SGBaselineAgent`(无 HTTP 客户端,直用 `arc_agi.Arcade` SDK)
- `scripts/train_sg.py` — 单游戏训练 + 终态 scorecard 打印

### 与原 SG 的差异
| 项 | 原 SG | 我们 |
|---|---|---|
| API 接口 | 自带 HTTP 客户端打 `three.arcprize.org` | 直接用 `arc_agi.EnvironmentWrapper.step()` |
| FrameData.score | 用 legacy `score:int` 跟踪进度 | 改用 `levels_completed:int` |
| available_actions | 旧 schema 没有 | 用 `list[int]` mask logits(动作过滤前置) |
| Tensorboard | 默认开启 | 暂禁(后续可加) |
| 设备 | CUDA-first | CPU 兜底,通过 `--device cuda` 切 GPU |

### 已验证(CPU 上 500 步 sanity)
- ✅ 模型 forward/backward 正确(`(1,16,64,64)` → `(1,4101)` logits)
- ✅ frame → tensor 转换:取最后子帧,one-hot 16 色
- ✅ ACTION6 坐标 0-63 范围内
- ✅ `available_actions` mask 正确(cn04 不可用 ACTION7,沃我们没派出过)
- ✅ Level transition 检测(`-1 → 0` 在 reset 时触发)
- ✅ 哈希去重(MD5 of frame+action_idx)
- ✅ GAME_OVER → 自动 RESET 续训(scorecard 显示 6 个 reset)
- ✅ Scorecard close + per-run 指标

### sanity-run 数据(cn04, 500 steps, CPU, seed=42)
```
loss:  0.641 → 0.385 → 0.302 → ~0.43 平台噪声
acc:   0.92  → 0.89  → 0.86 → ~0.85
buf:   0     → 427 unique 经验(ratio 85.4%,去重表现良好)
pos/neg ratio: 18:1 → 9:1 → 3.5:1(neg 信号随时间增加)
runtime: 450s 总,等价 1.1-1.2 fps(CPU 上 train_step 是瓶颈)
levels: 0/6(随机点击没碰到关 1 触发条件,baseline 29 步,符合预期)
```

## 关键观察

**cn04 反馈极密集(95% 动作改帧)**:这意味着 SG 的"frame change prediction"目标本身**几乎无判别能力**——模型很容易学到"全输出 1"。冠军方案能拿 12.58% 主要靠**采样的随机性 + 协调性**(避免重复同样的死路径),不是靠分类精度。

**含义**:
1. CPU 训练完全不会"卡死",但**很难真触发关卡通过**——random click 在 64×64 上击中正确触发点的概率太低
2. 我们的下一个增量必须是 **dolphin-in-a-coma 的状态栏遮蔽 + 连通块优先级** 来约束 ACTION6 候选,光靠 SG 的 vanilla 不够

## 未验证(需 GPU + 8h/game 预算)

- ⏳ SG 能否在 cn04 通过 level 1(baseline 29 步;原 SG 在 preview 比赛通关 2/3 个游戏,但 cn04 可能不在那 3 个里)
- ⏳ 跨游戏稳定性
- ⏳ 12.58% 是否可复现(需全 25 demo + 调度 200 GPU-h)

## 下一步路线

按 `docs/baseline-routes.md` 的 Day-1 增强清单:
1. **取消 level 间 model reset**(SG TODO 注释自承的简化点) — 无成本测一下
2. **状态栏遮蔽 + 连通块优先级 tier**(借 dolphin) — 给 coord head logits 加先验,治冷启动
3. **state-graph dedup + GAME_OVER 黑名单**(借 BS) — 避免反复死同一个状态
4. 在 GPU 上跑全 25 demo,出对照 baseline 表

## 复现 sanity 跑

```bash
uv sync
echo "ARC_API_KEY=..." > .env
uv run python scripts/train_sg.py --game cn04 --steps 500 --seed 42 --log-every 50
```
