# Cloud GPU 部署指南

> 本地 WSL2 无 GPU,SG baseline 训练需要 GPU 才能在比赛预算内出真分。本文档是给云 GPU 实例配置 + 跑 benchmark 的一次性配方。

## 1. 推荐机型

| 机型 | 提供方 | $/h(参考) | 适合 |
|---|---|---|---|
| **NVIDIA Tesla P100 16GB** | Kaggle / Vast.ai | **$0**(Kaggle)/ $0.2–0.4 | **当前选择**:Kaggle 周配 30h 免费,Pascal compute_60 |
| NVIDIA L4 24GB | Lambda / GCP | $0.5–0.8 | 比 P100 快 2–3×,如果预算够 |
| NVIDIA A10 24GB | Vast.ai | $0.4–0.7 | 同 L4 档,价格敏感 |
| NVIDIA RTX 3090 24GB | Vast.ai | $0.3–0.5 | FP32 强,FP16 也好 |
| Colab Pro T4/L4 | Google | $10/月 | 起步可选,12h 上限 |

**模型很小**(34M 参数,~130MB FP32),不需要 A100/H100。**带宽和 CPU 数量** 比 GPU 算力更影响 SG(每 5 步训一次,batch 64,瓶颈在数据流)。

### P100 具体笔记
- **架构**:Pascal,compute capability **6.0**。CUDA 12.1 / 12.4 wheel 都兼容(`cu121` / `cu124`)。
- **FP16**:支持但**性能不优**(Pascal 没有 Volta+ 的 Tensor Core)。**用 FP32 即可**,我们 34M 参数 FP32 vRAM <2GB,16GB 富裕得离谱。
- **Kaggle Notebook P100**:免费 30h/周,但是**评测的目标环境**(无网络、`/kaggle/working` 只读 9GB)。**不建议在 Kaggle 跑长基准**,留给最终提交用;benchmark 走 Vast.ai/自托管 P100。
- **预期 fps**:本地 CPU 1.2 fps → P100 预计 **20-50 fps**(模型小 + batch 64 跑 BCE,主要 overhead 在 forward+backward)。500 步从 8min CPU 变 ~30s P100。
- **跑全 25 demo 1h/game**:25h × $0.3 = **~$7.5**(Vast.ai)。Kaggle 免费但要拆分成 ≤12h notebook,中断恢复就靠 `runs/<tag>/results.json` 的 checkpoint。

## 2. 实例上一次性 setup

```bash
# 1. 装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. 拉项目
git clone <你的仓库 URL> arc-agi-3-pre
cd arc-agi-3-pre

# 3. 装本地依赖(uv 默认按 pyproject.toml,会装 CPU torch)
uv sync

# 4. **覆盖** torch 为 CUDA 版(关键!)
#    P100 (Pascal compute_60):cu121 最稳;cu124 也行
#    L4/A10/3090 (Ampere/Ada):cu124 优先
uv pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121

# 5. 验证 CUDA 可用
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

# 6. 设 API key
echo "ARC_API_KEY=<你的 key>" > .env
```

## 3. 跑 SG baseline benchmark

### 3.1 单游戏验证(10 分钟,看 GPU 速度)

```bash
uv run python scripts/train_sg.py --game cn04 --device cuda --steps 2000 --log-every 100
```

**期望 fps**:CPU 1.2 → P100 **20-50** → L4/A10/3090 30-80 → A100 80-150。如果 GPU fps 不到 10,检查 batch size、cuda init、numpy↔tensor 拷贝、I/O 瓶颈。

### 3.2 全 25 demo 1 小时/游戏(总 ~25h GPU 时长)

```bash
uv run python scripts/benchmark.py --device cuda --budget-min 60 --tag sg-bench-v1 \
    > runs/sg-bench-v1.log 2>&1 &
tail -f runs/sg-bench-v1.log
```

**输出**:
- `runs/sg-bench-v1/results.json`:每游戏摘要(steps, levels_completed, wallclock)
- `runs/sg-bench-v1/<game_id>.steps.jsonl`:逐步日志(可后处理画图)
- `runs/sg-bench-v1/scorecard.json`:官方 scorecard,含 RHAE 总分

### 3.3 短跑(预算紧 / 想快看结果)

```bash
# 5 个游戏 × 15min,总 75min
uv run python scripts/benchmark.py --device cuda --budget-min 15 \
    --games cn04,ft09,m0r0,lp85,r11l --tag quick-look
```

## 4. 监控

- **远程实时**:`tail -f runs/<tag>.log` 看 step 节奏 / loss 曲线
- **本地拉回**:跑完 `scp -r user@host:arc-agi-3-pre/runs/<tag>/ ./runs/` 拷回结果分析
- **GPU 状态**:`watch -n 1 nvidia-smi` 看显存 / 利用率

## 5. 资源经济性

- **L4 $0.6/h × 25h = $15** 跑全 25 demo 一轮
- **第一次跑足够**:看哪些游戏拿到分(任何 levels_completed > 0 都是信号),后续可以只重跑高潜力游戏
- 别跑 8h/游戏:除非你已经看到该游戏 30min 内确实在进步

## 6. 注意事项

- ❌ **不要在云上 push commit 到主仓**:训练数据/scorecard 不入库,但权重文件可能误入
- ✅ Kaggle 评测**无网络**:云上跑出的模型权重得通过 Kaggle Dataset 上传,**这步还没写**(等用户决定走"先占位提交"路径再做)
- ⚠️ benchmark.py 的 checkpoint 是基于 `results.json`,中断重跑会跳过已完成游戏。但**训练 model state 没持久化**,新游戏会从随机权重开始。这与 SG 原版"per-game 独立"一致。

## 7. 出结果后下一步

跑完 benchmark 你会得到一份 `scorecard.json` 和总分。然后:

1. 对照 `docs/baseline-routes.md` 的 fallback 信号:
   - frame-change AUC 不收敛 → 换 dolphin 训练-free 路线
   - coord head 模式坍塌 → 接 connected-component 段级采样
   - 跨关卡都打不过 → 转 BS state-graph 思路
2. 若分数 > 0,启动 Kaggle 提交工程(把模型 + 代码打成 Kaggle Dataset + Notebook)
3. 若分数 = 0,先把 dolphin 思路落地(段级采样 + 状态栏 mask 升级版),再上 GPU 重跑
