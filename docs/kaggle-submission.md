# Kaggle 提交格式调研(C1a) — 最终版

> **状态:架构搞清楚了**。来自 [inversion 官方 StochasticGoose 示例](https://www.kaggle.com/code/inversion/arc3-sample-submission-stochastic-goose)(LB 0.25,V1,T4 x2 GPU,29s 运行,运行通过 Playwright 抓 iframe 拿到全文)。

## 0. TL;DR

**Kaggle 评测 ≠ OFFLINE 模式直跑**。它启了一个 **`gateway:8001` 的 HTTP 网关在同一 docker 网络里**,我们的 agent 用 `OperationMode.ONLINE` + `ARC_BASE_URL=http://gateway:8001/` 跟它说话。**和真 ARC API 同一套协议,只是 base URL 换了**。"No internet" 仅指公网不通,**局域网 gateway 始终可用**。

提交也不是写 submission.json。我们写一个 `Agent` 子类,网关跟踪分数,Kaggle 评分系统自动产 `submission.parquet`。

## 1. Kaggle 提供的资产(`/kaggle/input/competitions/arc-prize-2026-arc-agi-3/`)

| 路径 | 内容 |
|---|---|
| `arc_agi_3_wheels/` | `arc_agi-0.9.6-py3-none-any.whl`、`arcengine-0.9.3-...whl`、`pillow-12.1.1-*.whl` — 离线 pip install 用 |
| `ARC-AGI-3-Agents/` | **整套 runner 代码** — main.py + agents/ 包,我们把自己的 agent 挂进 `agents/templates/my_agent.py` |
| 其它(待 Kaggle Edit 端 `!ls -R` 验证) | **极可能含 `environment_files/` 私有 game .py**(否则 FORGE 0.39 无法实现);需要确认实际路径名 |

**关键不确定点**:私有 game .py 是放在 `/kaggle/input/competitions/...` 让 BFS 能 importlib 读到,还是只塞在 gateway docker image 内部 BFS 不可达?**只有去 Kaggle Edit 跑 `!ls -R /kaggle/input/ 2>/dev/null | head -100` 才能确认**。FORGE 既然能 0.39,前一种概率 >95%。

## 2. Sample notebook 结构(4 cells)

### Cell 1 — pip install SDK 离线 wheel

```python
!pip install --no-index --find-links \
    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \
    arc-agi python-dotenv
```

注:**装的是 arc-agi 0.9.6**;本地我们装的是 0.9.8,有可能 API 漂移。要把 `requirements-kaggle.txt` 锁到 0.9.6 同步测一遍。

### Cell 2 — `%%writefile /kaggle/working/my_agent.py`

把 agent 类源码写进 `/kaggle/working/my_agent.py`。继承自 `from agents.agent import Agent`,实现:

```python
class MyAgent(Agent):
    MAX_ACTIONS = float('inf')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.game_id, self.frames, self.action_counter 由基类提供
        self.start_time = time.time()
        # ...

    def is_done(self, frames, latest_frame) -> bool:
        return any([
            latest_frame.state is GameState.WIN,
            (time.time() - self.start_time) >= 8 * 3600 - 5 * 60,  # 7h55m 单游戏硬上限
        ])

    def choose_action(self, frames, latest_frame) -> GameAction:
        # latest_frame.state (GameState.NOT_PLAYED / GAME_OVER / NOT_FINISHED / WIN)
        # latest_frame.levels_completed (int)
        # latest_frame.available_actions (raw ints [1..6] in gateway mode!)
        # latest_frame.frame (HxW int array)
        # 返回 GameAction;ACTION6 要 .set_data({"x": int, "y": int})
        # 可选: action.reasoning = "..."
        ...
```

### Cell 3 — 仅在 `KAGGLE_IS_COMPETITION_RERUN` 才跑的部分(真正提交时执行)

```python
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # 1) 等 gateway 起来
    !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \
          --retry-max-time 600 http://gateway:8001/api/games

    # 2) Runner repo 拷到可写位置
    !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \
           /kaggle/working/ARC-AGI-3-Agents

    # 3) 把自己的 agent 文件塞进去
    !cp /kaggle/working/my_agent.py \
        /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

    # 4) 重写 agents/__init__.py(原版 eager-import 一堆依赖如 langgraph,会炸)
    with open('.../ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
        f.write("""...AVAILABLE_AGENTS = {"random": Random, "myagent": MyAgent}""")

    # 5) 写 .env 指向 gateway(!! 关键 !!)
    with open('.../ARC-AGI-3-Agents/.env', 'w') as f:
        f.write("""SCHEME=http
HOST=gateway
PORT=8001
ARC_API_KEY=test-key-123
ARC_BASE_URL=http://gateway:8001/
OPERATION_MODE=online
ENVIRONMENTS_DIR=
RECORDINGS_DIR=/kaggle/working/server_recording
""")

    # 6) 跑
    !cd /kaggle/working/ARC-AGI-3-Agents && \
        MPLBACKEND=agg python main.py --agent myagent
```

### Cell 4 — 非 rerun(Edit 模式)写假 submission.parquet

```python
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    import pandas as pd
    submission = pd.DataFrame(
        data=[['1_0', '1', True, 1]],
        columns=['row_id', 'game_id', 'end_of_game', 'score'])
    submission.to_parquet('/kaggle/working/submission.parquet', index=False)
```

**结论**:submission.parquet 在 Edit 模式只是占位文件用来让 Kaggle 接受 notebook 提交。**真正的评分**:rerun 模式下,gateway 自己跟踪 scorecard,Kaggle 的评分服务从 gateway 读分,自动覆盖 submission.parquet。**我们不直接写实际分到 parquet**。

## 3. 跟我们 baseline-2 (BFS) 对接的 3 个关键改动

1. **抛弃 `bfs_benchmark.py` 的 Arcade harness,改写成 Agent 子类**。Logic 一样,但容器从 "main loop with env.step" 换成 "choose_action callback"。
2. **状态机更复杂**:choose_action 第一次调用时需要 BFS 解整局,之后逐步返回 action。需要 `self._solved_actions: deque[(action_id, data)]`。
3. **levels_completed 变化时需要重新 BFS**(下一个 level 的解算)。

## 4. 跟我们 baseline-1 (SG) 对接

幸运:**StochasticGoose sample notebook 就是 baseline-1 的近亲**。可以基本原样改成 baseline-1 当 fallback,在 BFS 没解出来的 game 上替换。

## 5. 几个细节坑(从源码读出)

- `latest_frame.available_actions` 在 gateway 模式下返回 **raw int(1..6)**,不是 `GameAction` enum。要 `getattr(action, "value", int(action))` 兼容。
- 原版 `agents/__init__.py` eager-import 多个 LLM agent (langgraph/smolagents),Kaggle 没装会炸。**必须重写 __init__**。
- agent 的 ACTION6 数据是 `action.set_data({"x": int, "y": int})`(注意是 set_data 不是构造参数)。
- 时间限制:**单 game 默认 8 小时**(我们看 `_has_time_elapsed = (time.time() - start) >= 8*3600 - 5*60`)。25 demo + ~3 hidden batches × 8h = 远超 12h notebook 总上限,**所以实际单 game 时间会被运行框架限制得多**(看不到具体值,需要看 ARC-AGI-3-Agents/main.py)。

## 6. 修正后的 C1b 工作量

| 工序 | 估时 |
|---|---|
| 拉 ARC-AGI-3-Agents repo 读 main.py + Agent 基类 | 30 min |
| 把 BFSSolver 包成 `BfsAgent(Agent)` 子类 | 1.5h |
| 注入 SG fallback for BFS-fail | 1h |
| 本地 mock gateway 联调(arc-agi-3-pre/server.py 应该可以拿来当 mock) | 1.5h |
| 写 submit_bfs.ipynb(直接照搬 sample 4 cell 结构) | 30 min |
| Kaggle Edit 验证 + 私有提交 1 次 | 1h(含等运行) |
| **合计** | **6h** |

## 7. 跟 baseline-2 现有 RHAE = 0.014 的关系

- 0.014 是在 demo 25 game 上 BFS 36% L0 通 + 0% L1 通的成绩。
- StochasticGoose sample 在 Kaggle 私评是 **0.25**。它就是我们的 baseline-1 同款 SG 架构(我们的 baseline-1 demo 上 0.003 也很差)。
- 这说明 **demo set 和 Kaggle 私评集不可直接对比**,差异巨大。
- 真实预期:**先把现有 BFS 干净地提一次,看 LB 真分**。再决定优化方向。
