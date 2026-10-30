# 运行与下载

## 1. 环境

建议 Python 3.13，当前复现依赖的 MuJoCo / NumPy / SciPy 版本锁定在 `requirements-r29.txt`。
R29 不依赖 CUDA、PyTorch 或 OMPL。不要把固定前缀重放入口当作任意物体随机测试入口。

Windows PowerShell：

```powershell
git clone https://github.com/Gabric-cau/ee5062-kuka-sharpa.git
cd ee5062-kuka-sharpa
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-r29.txt
```

macOS / Linux：

```bash
git clone https://github.com/Gabric-cau/ee5062-kuka-sharpa.git
cd ee5062-kuka-sharpa
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-r29.txt
```

## 2. 运行数据

从本仓库 `handoff-20261030` Release 下载 `r29-runtime-inputs.zip`，放到项目目录。
安装器先验证全部成员的 SHA-256，再安装到原路径；如已有不同内容则拒绝覆盖。
也可以用已登录 GitHub CLI 下载：

```bash
gh release download handoff-20261030 --repo Gabric-cau/ee5062-kuka-sharpa --pattern r29-runtime-inputs.zip
```

Windows：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_r29_inputs.py --archive r29-runtime-inputs.zip
.\.venv\Scripts\python.exe scripts/prepare_r29_inputs.py --check
.\.venv\Scripts\python.exe -m pytest -q tests/test_prepare_r29_inputs.py tests/test_coordinated_pick_place_r29.py
.\.venv\Scripts\python.exe scripts/run_coordinated_pick_place_r29.py --output results/my_r29
```

macOS / Linux 对应命令将 Python 路径替换为 `.venv/bin/python`。输出目录必须不存在。
物理运行会生成约数百 MB 日志，内存开销明显大于压缩文件大小；先单任务运行，测量本机资源后再并发。
本次历史 R29 动态通过环境为 Linux，仓库不预先承诺 Windows 上的数值重放结果。
2026-10-30 新 Mac 实际运行在第 3,599 步、7.198 s 触发 `replay_mismatch`，
关节速度重放偏差约 0.0000123793 rad/s 超过 0.00001 rad/s 前缀匹配门；力和其他重放项均在门内。
保留了失败原始轨迹和结果，见 Release 的 `mac-reproduction-failure.zip`。
若本机出现同类结果，先提交环境与失败记录，由 lead 检查可移植性，避免直接放宽门限或伪报成功。

## 3. 独立核验与视频

```bash
.venv/bin/python scripts/audit_coordinated_pick_place_r29.py results/my_r29 --output results/my_r29/command_audit.json
.venv/bin/python scripts/audit_r29_task_sequence.py results/my_r29 --output results/my_r29/sequence_audit.json
.venv/bin/python scripts/render_coordinated_pick_place_r29.py --input results/my_r29 --output results/my_r29_video
```

Windows 使用 `.\.venv\Scripts\python.exe`，保持 `MUJOCO_GL` 未设置。
仅无显示的 Linux 渲染时使用 `MUJOCO_GL=egl`。渲染器重建记录状态，不重跑物理。
`r29-recorded-evidence.zip` 是历史实际运行的完整记录与视频，按项目相对路径解压后可独立审计。
输入安装器仍须执行，因为历史审计会检查原输入哈希。

前一版预提交 ZIP 仅保留本地，因包含实验室连接资料而不随公开 Release 发布；它也漏带 R29 运行前缀数据。
当前复现依据是本仓库 commit + `r29-runtime-inputs.zip`；课程最终提交包需从此重新组装。

## 4. 其他入口

工作台、工具、触觉、滚动见 [V7_RUNBOOK.md](../V7_RUNBOOK.md)。这些旧手册中指向未下载 `results/` 的链接是历史实验路径。
脚本入口可用 `--help` 查看参数。V7 部分入口需要 `PYTHONPATH=src`：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts/run_v7_workbench.py --view --duration 60 --output results/my_operator
```

真人操作后保留视频、`operator_events.json` 和 `action_log.json`；自动回放单独标记。
OMPL 避障历史路径使用 Linux `ompl==2.0.1`，Windows/WSL2 安装和执行尚需专门验证。
缺少 OMPL 时先完成不依赖它的任务，不将导入失败写成物理任务失败或避障成功。

