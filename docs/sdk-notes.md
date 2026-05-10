# ARC-AGI-3 SDK 摸底笔记

> 基于 `arc-agi==0.9.8` + `arcengine==0.9.3`(2026-05-10 装箱)。

## 1. 顶层结构

两个 pip 包,职责分明:

| 包 | 角色 | 关键导出 |
|---|---|---|
| `arcengine` | **游戏引擎本身**:回合制循环、64×64 渲染、动作枚举、数据模型 | `GameAction`、`FrameData`、`FrameDataRaw`、`GameState`、`ActionInput`、`ARCBaseGame`、`Camera`、`Sprite`、`Level` |
| `arc_agi` | **runtime / API 网关**:本地 wrapper、远程 wrapper、scorecard 管理、HTTP server | `Arcade`、`EnvironmentWrapper`、`LocalEnvironmentWrapper`、`RemoteEnvironmentWrapper`、`ScorecardManager`、`OperationMode` |

## 2. OperationMode(关键)

```python
class OperationMode(str, Enum):
    NORMAL      = "normal"       # 本地 + 远程 API(默认)
    ONLINE      = "online"       # 仅远程 API,需要 ARC_API_KEY
    OFFLINE     = "offline"      # 仅本地 environment_files/
    COMPETITION = "competition"  # 比赛模式(Kaggle 评测器用)
```

- 远程地址默认 `https://three.arcprize.org`,通过 `ARC_BASE_URL` 改写。
- `ARC_API_KEY` 从 https://three.arcprize.org/ 注册申请。无 key 调 `get_environments()` 返回 401。
- **OFFLINE 模式扫描** `environments_dir`(默认 `./environment_files`)下的 `<game_id>/<version>/metadata.json`。空目录就 0 个游戏。
- **比赛环境**(Kaggle 无网络)推断走 `COMPETITION` 模式,游戏由比赛镜像挂载好。

## 3. Arcade 对外 API

```python
arcade = Arcade(
    arc_api_key="",           # 或读 ARC_API_KEY
    arc_base_url="https://three.arcprize.org",
    operation_mode=OperationMode.NORMAL,
    environments_dir="environment_files",
    recordings_dir="recordings",
)

arcade.get_environments() -> list[EnvironmentInfo]
arcade.make(game_id, seed=0, scorecard_id=None,
            save_recording=False, include_frame_data=True,
            render_mode=None, renderer=None) -> EnvironmentWrapper
arcade.create_scorecard(source_url, tags, opaque) -> str  # card_id
arcade.open_scorecard(...) / close_scorecard(card_id) / get_scorecard(card_id)
arcade.listen_and_serve(host="0.0.0.0", port=8001, competition_mode=False)  # 启动 HTTP server
```

## 4. EnvironmentWrapper(每局游戏的句柄)

Gym 风格,`reset()` / `step(action, data, reasoning)`:

```python
env = arcade.make("ls20")
frame: FrameDataRaw | None = env.reset()
frame = env.step(GameAction.ACTION1)
frame = env.step(GameAction.ACTION6, data={"x": 32, "y": 32},
                 reasoning={"thought": "click center"})
```

Properties: `action_space`、`observation_space`、`info`。

## 5. GameAction(动作空间)

```python
GameAction.RESET    = 0  (SimpleAction)   # 重启:第一次 → full_reset;之后 → level_reset
GameAction.ACTION1..5 = 1..5 (SimpleAction)  # 无参,游戏自定义语义
GameAction.ACTION6  = 6  (ComplexAction)  # 参数 {x:0..63, y:0..63}
GameAction.ACTION7  = 7  (SimpleAction)   # 通常是 undo(支持的游戏)
```

工具方法:`is_simple()` / `is_complex()` / `set_data(dict)` / `from_id(int)` / `from_name(str)` / `all_simple()` / `all_complex()`。

`reasoning` 是不透明 JSON,会被原样回显;有 `MAX_REASONING_BYTES` 上限。

## 6. FrameData / FrameDataRaw(观察)

| 字段 | 类型 | 含义 |
|---|---|---|
| `game_id` | `str` | 当前游戏 id |
| `frame` | `list[list[list[int]]]` | **关键**:形状 `[N_subframes, 64, 64]`,每格是 0–15 的调色板索引(单次 step 可能产 1–N 帧) |
| `state` | `GameState` | `NOT_PLAYED` / `NOT_FINISHED` / `WIN` / `GAME_OVER` |
| `levels_completed` | `int` 0..254 | 已通关数 |
| `win_levels` | `int` 0..254 | 全部关卡数 |
| `action_input` | `ActionInput` | 上一步执行的动作回显 |
| `available_actions` | `list[int]` | 当前帧合法动作 id 列表 — **应用此过滤动作空间** |
| `guid` | `str?` | 本局唯一 id |
| `full_reset` | `bool` | 上一次 RESET 是否触发了 full_reset |

`FrameDataRaw` 与 `FrameData` 同字段,但内部 `frame` 用 `numpy.ndarray` 列表存(性能路径)。

## 7. 渲染流水线(引擎层 OVERVIEW.md 摘录)

- **画布固定 64×64,16 色**。摄像机自动放大并 letterbox。
- **回合制**:不接收输入则不前进。
- **每个动作产 1–N 帧**(渲染中间过程也算)。
- `RESET`:有动作记录 → `level_reset()`;否则 → `full_reset()`。
- `next_level()` 默认会 +1 关卡 +1 score,如果是最后一关则 `win()`。

## 8. Scorecard

```python
EnvironmentScorecard:
    card_id: str
    score: float                # 总得分
    environments: list[EnvironmentScoreList]
    tags_scores: list[EnvironmentScore]
    competition_mode: bool
    open_at / last_update: datetime
```

`Arcade.create_scorecard()` 返回 `card_id`,然后 `make(game_id, scorecard_id=card_id)` 把成绩归到这张卡。

## 9. 本地开发的现实

- **没有 ARC_API_KEY 就跑不动公测游戏**(401)。需要去 https://three.arcprize.org/ 申请。
- **arcengine 自带的只是引擎**,不带任何比赛游戏。仓库 `arcprize/ARCEngine` 有 `simple_maze`、`merge` 等示例游戏可拷过来当本地玩具。
- **官方 agent 仓** `arcprize/ARC-AGI-3-Agents` 用的是 HTTP 协议(`http://localhost:8001/api/games`),配合 `Arcade.listen_and_serve()` 使用。我们直接用 SDK in-process 更简单。

## 10. 后续要确认(开赛后)

- [ ] Kaggle 评测的 `OperationMode.COMPETITION` 具体行为(是否预挂载游戏到 `environment_files/`?)
- [ ] 提交格式:notebook 还是 wheel?最大运行时间?GPU 类型?
- [ ] 是否允许携带预训练权重(开源仓里要包含)?
