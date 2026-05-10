# 复现 StochasticGoose 12.58% 的两条路径

## 现状摸底

`ref/ARC3-solution/` 已克隆并初始化 submodule(pinned 提交 `5a258aa`)。结构:

```
ref/ARC3-solution/
├── Makefile                        # action / install / tensorboard / clean
├── README.md
├── requirements.txt                # numpy 2.3.2 + tensorboard + torch 2.8.0
├── custom_agent.py                 # 1 行 shim:from custom_agents.action import Action
├── custom_agents/
│   ├── __init__.py                 # 空
│   ├── action.py                   # 489 行,核心:ActionModel + Action(Agent)
│   └── view_utils.py               # tensorboard 可视化
├── utils.py                        # 76 行,实验目录 / 日志
└── ARC-AGI-3-Agents/               # 老版 submodule
    └── agents/
        ├── agent.py                # HTTP 客户端基类(用 requests + ARC_API_KEY)
        ├── structs.py              # 老版 FrameData(用 score 字段,无 levels_completed/win_levels/available_actions)
        └── ...
```

**关键不兼容**:SG 用的老版 `agents/structs.py.FrameData` 的字段结构与**当前 `arcengine.FrameData` 不一致**:
- 老版:`score: int`(单一分数字段)
- 新版:`levels_completed: int` + `win_levels: int` + `available_actions: list[int]`

老版 agent 走 **HTTP** 直接打 `https://three.arcprize.org`,把 JSON 反序列化进自己的 `FrameData`。如果服务端 JSON 已经 evolve 到新 schema,老版客户端会缺字段或字段名不对。

## Path α:直接跑老版仓(verbatim,先试)

最便宜的尝试。只需要 ARC_API_KEY,不用改一行代码。

```bash
cd ref/ARC3-solution
cd ARC-AGI-3-Agents && cp .env-example .env
# 把 ARC_API_KEY 填进 .env
cd ..

# README 第 4 步要做的两个 monkey-patch
# (1) ARC-AGI-3-Agents/agents/__init__.py 加:
#     import sys, os
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#     from custom_agent import *
# (2) ARC-AGI-3-Agents/agents/structs.py FrameData 加字段:
#     available_actions: list[GameAction] = Field(default_factory=list)

make install
make action
```

预期:训练日志 + tensorboard runs/。**如果老 client / 新 server 兼容,几小时后能复现接近 12.58%**。

**风险**:服务端 JSON 不向后兼容老版 schema → pydantic 校验报错,需要改 `agents/structs.py`。如果只是少了 `available_actions` 那加上就行;如果 `score` 改名了,要把整套指标统计改一遍。

## Path β:把 SG 核心搬进我们自己的 src/(modern SDK)

如果 α 跑不通(预计 50% 概率),工程量约 **~半天 ~ 1 天**:

1. 把 `custom_agents/action.py` 的 `ActionModel(nn.Module)`(网络) + `Action.train_model()` / `Action.choose_action()`(训练循环 + 推理) 提到 `src/agents/sg_baseline.py`
2. 替换数据接口:用 `arc_agi.Arcade` + `EnvironmentWrapper.step()` 取 `FrameDataRaw`,用新的 `frame: list[ndarray]` 而不是 `list[list[list[int]]]`
3. 把 SG 的 `score` 进度跟踪改成新版的 `levels_completed`
4. 用 `available_actions` 做动作 mask(SG 老版没这个,新版能直接用,等于免费增益)
5. 把 `view_utils.save_action_visualization` 这套 tensorboard 可视化照搬

## 路径决策(等 key 后即时拍板)

```
    拿到 ARC_API_KEY
          │
          ▼
   make install && make action
          │
   ┌──────┴──────┐
   ✓ 跑通       × 报错
   │             │
   走 α          看错误类型
   收 12.58% 数据  │
                ├─ pydantic 字段缺失 → 加字段重跑(轻改 α)
                └─ JSON 整体变了 → 走 β,从 src/agents/sg_baseline.py 起
```

## 跑通后要采集的量化数据

无论 α 还是 β,跑完都要拿到下面这些数,作为我们 baseline 改进的对照:
- 总 RHAE 分(SG 复现版)
- 每个游戏:用了多少 env-step、停在第几关
- frame-change AUC 曲线(随训练步)
- coord head 概率热图(检查是否 mode collapse)
- tensorboard 截图 ⇒ `runs/` 是 .gitignore 的,但选关键截图入 `docs/`

## 一行 wrapper

`scripts/run_sg.py` 包装 α 路径,要点:检查 .env、cd 到 ref/ARC3-solution、应用 monkey patch、调 make。见该文件。
