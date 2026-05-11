# Kaggle 提交操作步骤(Option B 并行双线)

> 给你跑的 step-by-step。代码这边都准备好了:`notebooks/submit_bfs.ipynb` + `notebooks/submit_sg.ipynb`。

## 0. 前置一次性工作

把 `arc-agi-3-pre` 仓库的当前状态上传成 **public Kaggle dataset**(一次性,后续每次代码更新走 "New Version")。

### 选项 A:从 GitHub URL(推荐)

1. Kaggle → Datasets → New Dataset → "Import from external source"
2. URL = `https://github.com/<your-user>/arc-agi-3-pre`(必须是 public repo)
3. Dataset title:`arc-agi-3-pre`(slug 会变成 `arc-agi-3-pre`)
4. License:Apache 2.0 或 MIT
5. Create

### 选项 B:本地打包上传

```bash
cd /mnt/d/code/Contest/kaggle
zip -r arc-agi-3-pre.zip arc-agi-3-pre/ \
  -x 'arc-agi-3-pre/.venv/*' \
  -x 'arc-agi-3-pre/runs/*' \
  -x 'arc-agi-3-pre/environment_files/*' \
  -x 'arc-agi-3-pre/__pycache__/*' \
  -x 'arc-agi-3-pre/.git/*' \
  -x 'arc-agi-3-pre/.playwright-mcp/*'
```

然后 Kaggle → Datasets → New Dataset → Upload → 选 zip。

**确认上传后的 dataset 路径**:在任何 notebook 里 `!ls /kaggle/input/` 看到的路径,通常是 `/kaggle/input/arc-agi-3-pre/`。**如果不是这个名字**,改 `notebooks/submit_*.ipynb` cell 2 的 `REPO_DATASET_CANDIDATES`。

## 1. 创建 BFS 提交 notebook

1. Kaggle → Code → 进入比赛 `arc-prize-2026-arc-agi-3` → "New Notebook"
2. Settings(右侧栏):
   - Accelerator: **None**(BFS 不需要 GPU,省 quota)
   - Internet: **Off**(私评强制 off,Edit 模式可以临时 on 调试)
   - Persistence: Files only
3. Add data:
   - 比赛数据 `arc-prize-2026-arc-agi-3` 应已自动挂载
   - 添加我们的 `arc-agi-3-pre` dataset
4. **替换全部 cell**:把本地 `notebooks/submit_bfs.ipynb` 的 4 个 cell 一个一个复制粘贴过去
5. 点 "Save Version" → "Save & Run All (Commit)"
6. Edit 模式跑完(应该几秒,只写 dummy submission.parquet),状态变 ✅
7. 右上角 "Submit to Competition" → 提交

## 2. 创建 SG 提交 notebook(同步进行)

1. 同样新建 notebook
2. Settings:
   - Accelerator: **T4 x2**(SG 需要 GPU,且 T4 比 P100 还快)
   - 其它同上
3. Add data:同 BFS
4. 复制粘贴 `notebooks/submit_sg.ipynb` 的 4 个 cell
5. Save & Run All Commit
6. Edit 模式应该 1-2 min(import torch + 检测 GPU)
7. Submit to Competition

## 3. 等评分(每个 notebook 评分约 8-12h)

Kaggle 评分 pipeline:
1. 重新拉一个 docker 镜像
2. 起 gateway:8001 服务(我们看不见,但每个调用都会被监听)
3. 跑我们的 notebook,设 `KAGGLE_IS_COMPETITION_RERUN=1`
4. notebook 调用 gateway 评测私评游戏
5. gateway 关 scorecard,Kaggle scorer 读分,生成最终 LB 分

**预期**:
- BFS notebook:LB **未知**,demo 上 RHAE = 0.014。可能 0.01-0.10。**主要目的是验证 pipeline 通**。
- SG notebook:LB **预期 0.20-0.30**(对标官方 SG sample 的 0.25)

## 4. 看分

- Submissions 页面看自己的分
- LB 页面看排名

## 5. 我们能在本地做的同步检查

trick:本地不知道你 Kaggle dataset slug 是什么,先用 `unused-but-needed-now.txt` 记录下来:
- 实际 dataset URL:_____________________(你填)
- 实际 `/kaggle/input/` 下的路径:_____________________
- BFS notebook 跑分:_____________________
- SG notebook 跑分:_____________________

## 6. 常见踩坑

| 症状 | 原因 | 解决 |
|---|---|---|
| `arc-agi-3-pre` not at `/kaggle/input/arc-agi-3-pre` | Kaggle 自动 slugify 时大小写不同 | `!ls /kaggle/input/` 确认实际名字,改 `REPO_DATASET_CANDIDATES` |
| `ModuleNotFoundError: src` | 没把 repo 加到 `sys.path` | 已加,但要确认 cell 2 跑过 |
| `Connection refused: gateway:8001` 在 Edit 模式 | 正常,Edit 模式 gateway 不会起 | 看 cell 3 里的 `if KAGGLE_IS_COMPETITION_RERUN` |
| 评分跑了 12h 自动 kill | 超时 | 降低 budget(BFS 减 `--max-levels`,SG 减 `--budget-min`) |
| 评分跑完 LB 还是 0 | gateway 接口对不上 | 看 Logs 标签,搜 "ERROR" |

## 7. 我之后要做的事

等你两个分回来,我会:
1. 把 LB 分填到 `docs/baseline-2-forge-bfs.md` 和 `docs/baseline-1-sg-port.md` 的"实际成绩"段
2. 根据两个分的差距,确定 Day-4 是优先做 hybrid(BFS+SG)还是单独优化 SG
3. 调研 LB 上 0.39+ 的更厉害方案,做下一轮迭代
