# Baseline 路线对比与推荐

> 输入:30 天 preview 比赛冠/亚/季军三份开源方案 + 技术报告 + 一篇 arxiv 论文。
> 调研时间:2026-05-10。

## 0. 评分公式(精确)

每关:`S_{l,e} = min(1, h_{l,e} / a_{l,e})²`,其中 `h` = 人类动作数(取**第二少**,outlier-robust),`a` = agent 动作数。

每个 env(线性加权,后面的关卡更值钱):`E_e = Σ_l (l · S_{l,e}) / (n(n+1)/2)`

总分:`T = (1/|D|) · Σ_e E_e`

**含义**:平方项 + 后面关卡加权 ⇒ **能完整通关一两个游戏 ≫ 在 N 个游戏第一关磕磕绊绊**。这是为什么 SG 的"per-level 死磕"打法比 BS 的"通用价值网络"得分更高。

## 1. 数据集形状

- 公开榜:**25 demo + 55 半私 + 55 全私 = 135 个 env**,每个 5–6 关。
- 人类基线:每 env ≥10 人,只保留"≥2 人首次尝试通关"的 env。
- 前沿 LLM(2026-03):Gemini 3.1 Pro Preview 0.37% / GPT-5.4 0.26% / Opus 4.6 0.25% / Grok-4.20 0.00%。
- 系统 prompt 一行话,无工具。

## 2. 路线 A — CNN 帧变化预测(StochasticGoose,1st 12.58%)

**仓**:<https://github.com/DriesSmit/ARC3-solution>(`custom_agents/action.py`)
**博客**:<https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db>

### 核心思想
把每一关当成独立任务,**在线**训一个二分类 CNN:给 (state, action),预测"这一步会不会改变下一帧"。无任何离线数据、无搜索,纯随机采样按概率分布选动作。

### 网络
- 输入:64×64 → 16 通道 one-hot
- Backbone:Conv2d 16→32→64→128→256(4 层)
- Action head:MaxPool → 512 → 5 logits(对应 ACTION1–5)
- Coord head:再 4 层 Conv2d 256→…→32→1 → 4096 logits(每像素一个)

### 训练
- Loss:在被采到的那个 logit 上做 BCE,标签是"帧是否变了" {0,1};加 action-/coord-entropy 正则
- Adam lr=1e-4 batch=64,**每 5 个 env step 训一次**
- 经验池 deque maxlen=200k,key = MD5(frame_bytes + action_idx) 去重
- **进入新关卡时模型和经验池一起 reset**(重要简化,留给我们改进空间)

### 推理
sigmoid → 把 5 + 4096 = 4101 logits 拼起来,coord 那 4096 个除以 4096 保持 simple action 公平,归一化后**采样**。idx>4 ⇒ ACTION6(y, x)。

### 战绩与代价
- 12.58%、18 关、**完整通关 2/3 个游戏**
- 单 GPU、~100k env-step / game / 8h 预算
- 冷启动会浪费 ~350 次乱点(blog 自承)

---

## 3. 路线 B — State Graph + ResNet18 价值网(Blind Squirrel,2nd 6.71%)

**仓**:<https://github.com/wd13ca/ARC-AGI-3-Agents/tree/blindsquirrel>(`agents/blind_squirrel.py`)

### 核心思想
把每关当成确定性 MDP,**显式建状态图**,通关后用图最短距离训一个 value net,后续关卡用它给动作排序。

### 状态去重
`hash((game_id, score, frame))`,frame 是原始网格元组 — **精确哈希,无 embedding**。

### 动作空间压缩(Valid-Actions Model,规则)
对每帧做**同色连通块分割**(按钮),在按钮上点击 ≡ 在按钮内任一像素点击,把 4096 个点击候选压成 ~20–50 个。每个按钮带规则计算的 `rweight`,扫到没改变帧的按钮就把它的 rweight 清零。**作者明说这块是最该用神经网替代的部分**。

### 图扩展
对 rule-weighted valid action 做 ε-greedy。已访问转移走缓存(图 = 经验池本体)。loop / no-change → 该按钮 rweight 归零;GAME_OVER → 标记终态避开。

### Value Net
- 预训练 ResNet-18,把 conv1/bn1/relu/maxpool 换成 `Embedding(16,16)` → Conv2d(16→64) → BN+ReLU(因为输入是 0–15 调色板)
- Head:state(512→64) ⊕ action(26 dim → 64) → cat → 128→64→1
- Action 26 dim = 6(action 类型 one-hot) + 16(颜色) + 1(规则性) + 1(尺寸) + 1(y) + 1(x)

### 训练
- 触发:每次 score+1 时
- 标签:从赢的状态 BFS 倒推每个状态的距离,advantage 化:`MODEL_SCORE_MAG · (state_d − future_d) / max_d`,MSE
- Adam lr=1e-4 batch=32 10 epoch,15 分钟封顶

### 推理
ε-greedy on `model_value × rweight`;首关或 ε 命中走纯 rweight 采样。

### 战绩与代价
6.71% / 13 关 / 通关 1 个游戏。代码自承"赛前几天才动手",留有大量可优化点。

---

## 4. 路线 C — 训练-free 图探索(dolphin-in-a-coma,3rd 3.64% → 17/25 修复后)

**仓**:<https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore>(`--agent=heuristicagent`)
**论文**:<https://arxiv.org/abs/2512.24156>

### 核心思想
**零神经网络、零训练**。帧处理:连通块分割 + **遮蔽状态栏区域** + 按尺寸/颜色突显度分 **5 个优先级**。算法 1 = 层级动作选择:**先在所有图节点穷尽 priority k 的所有未试动作,再下到 k+1**;到边界状态走图最短路径。

### 战绩与代价
median 17/25(post-bugfix),与亚军同档。零 GPU、零训练数据。**对状态栏布局非标的游戏脆弱,对随机性/部分可观测无解**。

---

## 5. 三路线对比

| 维度 | A. CNN + RL | B. Graph + ResNet | C. 训练-free 图探索 |
|---|---|---|---|
| 已验证战绩 | **12.58%** 🥇 | 6.71% | 3.64% / 17-25% |
| 通关游戏数 | 2 | 1 | 0(平均关卡多) |
| GPU 需求 | 1 张 | 1 张 | 无 |
| 训练数据 | self-play 在线 | self-play 在线 | 无 |
| 实现复杂度 | 低(~1 文件) | 中 | 中 |
| 与 SDK 对接 | 直接用 frame+available_actions | 需要写 connected-component segmenter | 同 B 但更简单 |
| 上限信号 | 高(冠军已证) | 中 | 中(会被 100 分天花板限制) |
| 主要风险 | 冷启动 350 步浪费;mode collapse | 规则 valid-actions 是脆弱点 | 状态栏布局变化即崩 |

## 6. 推荐与实施顺序

### 推荐:**A 为主线,Day 1 就吸收 B/C 的关键 idea**

理由:
1. RHAE 的平方+末关加权特性**强烈奖励"通关"** — A 是唯一证明能通关的方案
2. A 代码量最小,迭代快
3. A 的"frame-change prediction"目标天然契合 env 反馈结构(只有像素差),不需要外部信号
4. A 的两个明显改进点(level 间不 reset、加 state-graph dedup)就能进一步抬分

### Day 1 增强项(基于三家拆解)
1. **取消 level 间 reset**(SG 的明示简化点):跨关卡迁移参数 + 经验池
2. **叠加 state-graph dedup**(借 BS):用 `hash((score, frame))` 去重,避免 CNN 反复学同一个状态-动作
3. **状态栏遮蔽 + 连通块优先级 tier**(借 dolphin):作为 coord-head logits 的先验,治"350 步浪费"
4. **GAME_OVER 标记 + 避免**(借 BS):入图为终态,推理时设负无穷 mask
5. **按 `available_actions` 过滤 logits**(SDK 已暴露,SG 没充分用)

### Phase 5+ 实施拆分(不在本轮做)
1. `src/agents/sg_baseline.py`:照搬 SG ActionModel(单文件复刻)
2. `src/state/graph.py`:状态哈希 + 转移图 + GAME_OVER mask
3. `src/perception/segmenter.py`:连通块分割 + 状态栏遮蔽 + tier 打分(NumPy 即可)
4. `src/agents/sg_plus.py`:SG + 上述三项增强
5. eval harness:对每个公开 env 跑固定预算,出 RHAE-like 自评分

### Fallback 信号(切换到 C / B 的触发条件)
- A 在某个 env 的 frame-change AUC 20k 步内仍 ≈ 0.5(没学到):**降级到 C 训练-free 探索**
- coord head 收敛到几个热点像素(mode collapse):**改用 B 的 shape-button 离散动作空间**
- 跨游戏第一关都打不过:**瓶颈不在 perception 而在 cross-task 泛化,转向 B 的 value net 思路或更重的 world model**

---

## 7. 待办与开赛后核对项
- [ ] 拿到 ARC_API_KEY,跑一遍 quickstart 确认 SDK 行为与文档一致
- [ ] 跑 SG 原版仓 `DriesSmit/ARC3-solution` 看 12.58% 能否复现(可能需要整改输入接口)
- [ ] 开赛时核对 `OperationMode.COMPETITION` 的具体行为,确认 Kaggle 镜像内 environment_files 怎么挂载
- [ ] 核对预训练权重携带规则(BS 用了 ResNet18 预训练,如果禁用就只能 from-scratch)
- [ ] 核对算力上限(GPU 类型 / 时长 / 内存)
