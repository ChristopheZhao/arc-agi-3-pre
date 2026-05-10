# Baseline 2:FORGE-style BFS solver(候选之一)

> 与 baseline-1(SG-port + click prior)对应路线 A 不同,这里走 `docs/baseline-routes.md` 路线 B。**关键洞察:游戏不是黑盒**,可以 importlib 加载游戏类、deepcopy 状态、内存 BFS 找完美解。
>
> 状态:Infrastructure 通过本地验证,**1/4 抽样游戏 lvl 0 BFS 解出**,其它 3 个揭示具体短板。

## 0. 一句话定位

**离线 BFS perfect solver**:把 `environment_files/<game>/<version>/<game>.py` 当成可导入模块,实例化游戏类,在内存里 deepcopy 走子,搜出通关 action 序列后回放给 server。无需 RL,无需 GPU。

## 1. 关键发现 — 比赛不是黑盒

我们在 baseline-1 时假设只能 `env.step` 黑盒交互。但 quickstart 时观察到:

```
Successfully downloaded game cn04 (version: 2fe56bfb) to environment_files/cn04/2fe56bfb
Successfully loaded game class Cn04 from environment_files/cn04/2fe56bfb/cn04.py
```

游戏源码 obfuscated(类名随机字母)但**完全可执行**。FORGE 的 0.39 就是基于此:

```python
spec = importlib.util.spec_from_file_location('mod', 'cn04.py')
spec.loader.exec_module(mod)
GameCls = mod.Cn04
g = GameCls()
g.set_level(0)
g.perform_action(ActionInput(id=GameAction.RESET))  # in-process, no API
g2 = copy.deepcopy(g); g2.perform_action(...)        # 状态 fork,BFS 用
```

## 2. 数据流

```
                      Arcade SDK
                      ↓ (download .py)
              environment_files/<game>/<v>/<game>.py
                      ↓ (importlib + deepcopy)
              ┌─ BFSSolver (offline, in-memory) ─────────┐
              │                                          │
              │  load() → self.game_cls = GameCls       │
              │                                          │
              │  solve_level(L):                         │
              │    1. fresh game @ level L (2 RESETs)    │
              │    2. _scan_actions:                      │
              │         - try ACTION1-5 each, 看 diff>0   │
              │         - try ACTION6 step-2 网格点击,     │
              │           按 post-effect frame hash 去重   │
              │    3. (no actions?) warmup unlock:        │
              │         - 试 ACTION1-4 一个,re-scan      │
              │    4. BFS:                                │
              │         queue: (game_state, history)     │
              │         visited: set(state_hash)          │
              │         expand 每个 action,deepcopy step  │
              │         win = r.levels_completed > L      │
              │             or g._current_level_index > L │
              │                                          │
              │  state_hash:                             │
              │    "frame":  MD5(frame.tobytes())[:16]   │
              │    "full":   MD5(cloudpickle.dumps(g))    │
              │              ~3 ms/call,捕捉 sprite 状态 │
              │                                          │
              │  返回 BFSResult(actions | None, ...)      │
              └──────────────────────────────────────────┘
                      ↓ (replay action sequence)
              env.step(action_n, data_n) → server
```

## 3. 与 baseline-1 (SG-port) 对照

| 维度 | Baseline-1 (SG) | Baseline-2 (BFS) |
|---|---|---|
| 主算法 | CNN bandit (在线 RL) | **离线 BFS perfect solver** |
| 是否用游戏源码 | 否(黑盒) | **是,白盒搜索** |
| GPU 需求 | 必须 | **无,纯 CPU** |
| 算法天花板(单游戏) | 110 步通关 lvl 0(cn04) | 3 步通关 lvl 0(r11l) |
| 卡死场景 | lvl-2+ 普遍卡 | 隐藏 sprite 状态 + 狙击点击游戏 |
| Kaggle LB 参考 | StochasticGoose 0.25,SG++ 0.32 | **FORGE 0.39** |

## 4. 实测结果(本地,2026-05-10)

抽样 4 个 demo 游戏跑 lvl 0,30s budget per game:

| Game | 结果 | 步数 | Explored | Visited | 时间 | 备注 |
|---|---|---|---|---|---|---|
| **r11l** | ✅ solved | **3** | 2782 | 376 | 29.5s | BFS 完美场景 |
| cn04 | ❌ timeout | - | 35987(120s) | 9841 | 120s | frame hash 不够 |
| cn04 (full hash) | ❌ timeout | - | 16436(120s) | 8042 | 120s | cloudpickle 也不够 |
| m0r0 | ❌ timeout | - | 2836 | 876 | 32s | 同 cn04 |
| lp85 | ⚠️ no_actions | - | 0 | 0 | 2.4s | only ACTION6,任意点击 0 反馈 |

**solve rate = 1/4 ≈ 25%**。如果 25 demo 全集类似分布,纯 BFS 大概能覆盖 ~6-8 game。

## 5. 三类失败模式 + 对症药方

### A. 隐藏 sprite 状态(cn04, m0r0)
- **症状**:frame hash 把"看上去一样但实际不同"的状态去重,BFS 永远找不到 win
- **诊断**:cn04 的 5 个 obfuscated scalar 字段 ACTION1 后**都不变**,变化在 sprite 对象里
- **FORGE v18 没破**:它的 `_probe_hidden_fields` 只查 scalar,对 sprite 状态无效
- **可能解**:启发式搜索(A* + pixel-diff heuristic)、或接 CNN fallback

### B. 狙击点击游戏(lp85)
- **症状**:`available_actions=[6]` only,scan_step=2 采到的所有点击 diff=0
- **诊断**:游戏要求点击**特定小目标**(可能 1-2 像素),才能解锁后续
- **可能解**:dense scan(step=1,4096 candidates,~12s)、或先用 connected-component segmentation 找候选目标

### C. 状态空间太大
- **不在 cn04 这一档**(cn04 失败因 hash 不准,不是状态多)
- **真实 case**:可能存在某些游戏 BFS depth 30 内根本搜不到 win,需要 IDA* 或 RL 引导

## 6. 已实现 vs FORGE v18 对比

| 特性 | 我们 | FORGE v18 |
|---|---|---|
| importlib 加载游戏 | ✅ | ✅ |
| deepcopy + perform_action 搜索 | ✅ | ✅ |
| Frame hash | ✅ | ✅ |
| Full state hash (cloudpickle) | ✅ (新增) | ❌ (用 scalar 字段附加) |
| Effective action 扫描 | ✅ | ✅ |
| Click 效果去重 | ✅ | ✅ |
| Warmup unlock | ✅ | ✅ |
| Hidden field probing | ❌ | ✅ |
| Cross-level solution transfer | ❌ | ✅ |
| Counter A* (启发式) | ❌ | ✅ |
| CNN fallback | ❌ | ✅ |
| ACMD trigger finder | ❌ | ✅ |
| CLTI(BFS L0 demos → CNN) | ❌ | ✅ |

## 7. Day-3 / Day-4 优先级

**Day 3(明天)**:
1. **Dense scan mode**(step=1,~12s,治 lp85 这类)
2. **Cross-level transfer**(lvl N→N+1 解 + object centroid offset,FORGE 的关键 trick)
3. **跑全 25 demo 测 solve rate**(确认 25% 是真实分布)
4. 把 BFS solver 包到 scripts/benchmark.py 里出 RHAE 估算

**Day 4**:
1. CNN fallback wiring(用 baseline-1 的 SG-port 当 BFS-fail 的备份)
2. Hidden field probing(便宜,FORGE 也有,可能多救 1-2 个 game)

**Day 5-7**:
1. 学习成分增强:learned action ordering(value function for state expansion)
2. 调研 Kaggle LB 0.46+ 衍生方案(Jonathan Chan, ashvin singh)在 FORGE 基础上加了什么
3. 上 Kaggle 真提交

## 8. 已知未解 + 决策依据

- **cn04 / m0r0** 这类 sprite 隐藏状态游戏,纯 BFS 可能永远搞不定。这部分得靠 CNN fallback(SG-port 改进版,或学习 hidden state 的 representation)
- **是否要 GPU**:Day-3/4 做 BFS 增强 + benchmark 期间不需要 GPU。**Day 5+ 一定需要**(CNN fallback / value function 学习)
- **Kaggle LB 真分预期**:25% solve rate × 平均每解 ~20 步 vs baseline 50 步 → RHAE 估 0.06-0.15。比 baseline-1 的 0.003 量级跨越大,但仍距 0.39 有距离。**距离来自:更多 game 解出 + cross-level transfer + CNN fallback**。

## 9. 复现

```bash
uv sync  # 包含 cloudpickle
uv run python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
from src.agents.bfs_solver import find_game_source_and_class, BFSSolver
src, cls = find_game_source_and_class('r11l')   # 或任何 demo game
solver = BFSSolver(src, cls, scan_timeout=2, bfs_timeout=30)
solver.load()
res = solver.solve_level(0)
print(res)
"
```
