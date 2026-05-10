# Kaggle Notebook Benchmark Setup(P100)

> 你的 GPU 在 Kaggle 上 → 不需要额外环境。Edit 模式有网,可以照常调 ARC API 跑 benchmark。

## 前置认知

| 概念 | 说明 |
|---|---|
| **Edit 模式** | 你打开 notebook 在线编辑,**默认有网**,可以 `!pip install` 与 `requests`。**这是我们 benchmark 用的模式**。 |
| **Save & Run All / Submission** | 只在最终提交比赛时才走。**强制无网**,本文不涉及。 |
| **/kaggle/working** | 9GB 工作目录,session 之间不会持久化(下次开 notebook 是空的)。所有产出要 zip 下载。 |
| **/kaggle/input** | Dataset 挂载点,只读。 |
| **Secrets** | Add-ons → Secrets,把 `ARC_API_KEY` 等敏感字段以 key 形式存,代码里通过 `UserSecretsClient` 读。**不会出现在保存的 .ipynb 里**。 |
| **GPU 配额** | 免费版 30h/周;每个 session 12h cap;断网/超时会回收 working 目录。 |

## 一次性 setup(~10 分钟)

### 1. push 仓库到 GitHub

```bash
# 在本地 /mnt/d/code/Contest/kaggle/arc-agi-3-pre/
gh auth login                                                   # 如果没登录
gh repo create arc-agi-3-pre --private --source=. --remote=origin --push
# 或者已有 GitHub remote:
git push -u origin main
```

> 想保密就 `--private`,后面在 Kaggle 用 token 拉。
> 不在意公开就 `--public`,跳过 GITHUB_TOKEN 配置(简单)。

### 2. 在 Kaggle 创建 Notebook

1. 浏览器打开 <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3>
2. 顶部 "Code" 标签 → **"New Notebook"**(确保是从竞赛页进的,这样比赛规则会自动 attach)
3. 右侧 "Notebook options":
   - **Accelerator**: `GPU P100`
   - **Internet**: `Internet on`(默认就开,确认一下)
4. 顶部菜单 **File → Import Notebook → Upload** → 选 `notebooks/benchmark_p100.ipynb`

### 3. 配 Kaggle Secrets

右上角 **Add-ons → Secrets**,点 "Add Secret":

| Secret name | Value | 何时需要 |
|---|---|---|
| `ARC_API_KEY` | 你的 ARC key | 必填 |
| `GITHUB_TOKEN` | GitHub PAT(`repo` scope 即可) | 私库才需要 |

> 创 GitHub token:GitHub → Settings → Developer settings → Personal access tokens → Generate(classic 即可,选 `repo` 权限)。

### 4. 编辑 notebook 第一个 cell

把 `REPO_URL` 改成你的 GitHub URL,如果是私库把 `PRIVATE_REPO = True`。

### 5. Run All

菜单 **Run → Run All**,大约:
- Cell 1-2: <1 min(clone + 包检查)
- Cell 3 (smoke): ~3 min(2000 步 cn04 GPU)
- Cell 4 (benchmark): ~80 min(默认 5 game × 15min)
- Cell 5-6: <1 min(汇总 + zip)

**总 ~85 分钟,远低于 12h cap**。

### 6. 下载结果

跑完后右侧 "Data" / "Output" 面板会显示 `/kaggle/working/runs.zip`,点下载即可。

回到本地解压看 `runs/p100-short/results.json` 与 `scorecard.json`。

## 可选:跑全 25 demo

第 4 个 cell 改:
```bash
!python scripts/benchmark.py --device cuda --budget-min 60 --tag p100-full --log-every 200
```

预计 25h,**超过 Kaggle 12h cap**。两条解法:
- **拆 session**:跑一半,zip 下来,新建 session,把 zip 上传成 Dataset,挂载,继续(checkpoint 文件让 benchmark.py 自动跳过已完成游戏)
- **降预算**:`--budget-min 25` → 25 × 25 = 10.4h,刚好塞下

## 常见坑

| 现象 | 解决 |
|---|---|
| `cuda is_available() = False` | 右侧 Accelerator 选成了 "None",改 P100,重启 session |
| `pip install` 失败 | Internet 没开。右侧 toggle 打开重试 |
| `ARC_API_KEY length: 0` | Secret 没保存,或 key 名拼错 |
| benchmark 跑到一半 OOM | 不太可能(34M 模型),但若出现就 `--device cpu` 兜底(超慢) |
| 12h 到了被踢 | 看 `runs/<tag>/results.json` 已存了几个游戏,新 session 用同 tag 续跑会跳过 |
| `cudaErrorNoKernelImageForDevice` | Kaggle 预装的 torch 不支持 P100 的 sm_60。`requirements-kaggle.txt` 已固定 torch 2.5.1+cu121,确保 cell 3 跑过即可 |
| `SyntaxError: unterminated string literal` 在 cell 3 | 你把 `!python -c "..."` 拆成多行了。Jupyter 的 `!cmd` 是单行 magic,不能跨行 |
| 改了 `requirements-kaggle.txt` 但 Kaggle 没生效 | Kaggle 跑的是上传的 .ipynb 副本 + cell 1 fetch 出来的 repo 文件;notebook 本身需 re-import,repo 文件 cell 1 自动同步 |

## 跟自托管 P100 (Vast.ai) 的差异

| 项 | Kaggle P100 | Vast.ai P100 |
|---|---|---|
| 价格 | 免费(30h/周配额) | $0.2-0.4/h |
| 单 session | ≤12h | 任意 |
| 网络 | Edit 模式有 / Submission 无 | 一直有 |
| 持久存储 | 仅 Dataset(要传) | 整盘可写 |
| 适合 | 短验、提交时跑 | 长 benchmark 一气呵成 |
