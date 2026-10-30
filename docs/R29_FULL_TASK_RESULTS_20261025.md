# R29 完整抓取、搬运与放置

实验室 RTX4090 主机已完成一个真实 YCB 罐子的完整仿真任务，独立原生数据、
实际命令与任务顺序审计通过，`task_success=true`。范围是一个复用开发种子，
不是随机测试集，不代表 85% 成功率或整份课程作业完成。

## 直接查看与运行

- [完整双视角视频](../results/r29_native_video_review_20261025/trajectory.mp4)
- [搬运画面](../results/r29_native_video_review_20261025/mid.png)
- [撤手后的画面](../results/r29_native_video_review_20261025/final.png)
- [实际轨迹和触觉曲线](../results/r29_native_plot_20261024/trajectory_force.png)
- [原始结果](../results/r29_event_arrival_full_task_20261024/result.json)
- [独立命令与严格任务审计](../results/r29_event_arrival_full_task_20261024/independent_audit_macos_20261024.json)
- [Linux 独立审计](../results/r29_event_arrival_full_task_20261024/independent_audit_linux_20261025.json)
- [独立任务顺序审计](../results/r29_event_arrival_full_task_20261024/independent_sequence_audit_20261024.json)

在项目根目录只需运行这个完整案例入口，不需要逐个运行 R26/R27/R28：

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/run_coordinated_pick_place_r29.py \
  --output results/my_new_full_pick_place
```

输出目录必须是新目录。它从初始状态实际积分前 22 秒的已验证电机命令，随后
执行新的联合控制；不直接把物体设置到成功检查点。入口使用冻结的已知场景、
公共网格、旧 RGB-D 估计和已验证接近抓握前缀，不是任意物体抓姿生成器。

运行后可独立核验，并在 Linux/4090 上生成视频：

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/audit_coordinated_pick_place_r29.py \
  results/my_new_full_pick_place --output results/my_new_full_pick_place/audit.json
PYTHONPATH=src:scripts .venv/bin/python scripts/audit_r29_task_sequence.py \
  results/my_new_full_pick_place --output results/my_new_full_pick_place/sequence_audit.json
MUJOCO_GL=egl .venv/bin/python scripts/render_coordinated_pick_place_r29.py \
  --input results/my_new_full_pick_place --output results/my_new_full_pick_place_video
```

## 实测结果

新加坡时间 2026 年 10 月 24 日 23:49:02 至 23:53:42，监督进程正常退出，退出码 0。
81,277 个连续 2 ms 物理步覆盖 162.554 秒仿真，含 22 秒前缀和 140.554 秒后续控制。

| 指标 | 本次结果 | 原验收门 |
| --- | ---: | --- |
| 最大实际物体抬升 | 82.924 mm | 超过 70 mm |
| 搬运开始前连续合格持物 | 3.396 s | 至少 0.1 s |
| 搬运期间实际持物样本 | 22,792 / 22,792 | 离桌、手部承载 |
| 最终平面放置误差 | 4.160 mm | 小于 25 mm |
| 撤手完成后连续稳定释放 | 3.498 s | 至少 3 s |
| 最终所有手部几何与物体间隙 | 148.467 mm | 大于 100 mm |
| 全程通道峰值，原生积分缓存 | 5.451 N | 不超过 12 N |
| 全程通道峰值，独立刷新接触记录 | 5.421 N | 不超过 12 N |
| 外部目标 qpos / qvel 写入记录 | 0 / 0 | 0 |

严格核验还检查放置线速度、角速度、桌面支撑和机器人/桌面穿透不超过 2 mm；
它不证明任意障碍物避碰。写入计数是记录器断言，独立审计没有重新积分物理。
45 个冻结源码文件及输入哈希保持一致；70,277 条后续实际命令逐条通过关节、
速率、偏置和电机力矩约束复核。

在线监视器以初始目标高度计算 held 为 51.328 s；独立旧严格核验器以首个原生
样本高度计算为 51.348 s，两个基准的门结论均通过。上表另检查搬运前持物
和撤手后放置，避免错误的动作顺序被计为成功。

## 修复与物理边界

此前按预设时长结束试提，实际腕部仍在收敛。现在实际腕位误差小于 0.5 mm、
线速度小于 0.5 mm/s 连续保持 0.15 s 才切换阶段。每阶段均有有限超时，完整
执行试提、抬升、搬运、下降、支撑确认、松手、撤手和稳定检查。

机械臂与手指联合求解；接触法向速度包含腕部刚体运动补偿和压力误差修正，
求解结果是实际电机命令的唯一写入来源。初始 1.5g 静态分配仅给出三指压力
目标，不是在线重分配或动态承载证明。最终成功以实际运动和接触数据判定。

原生 MuJoCo 3.15.0 由 CPU 推进物理，RTX4090 用于 EGL 渲染，本次未训练神经网络。

| 编译模型参数 | 数值 |
| --- | --- |
| 重力 / 物理步长 | 9.81 m/s² / 0.002 s |
| 接触求解迭代数 | 100 |
| 目标质量 | 0.349 kg |
| 目标主惯量 | 0.000359131 / 0.000357194 / 0.000204549 kg·m² |
| 目标与桌面各自摩擦参数 | 0.6 / 0.005 / 0.0001 |
| 目标接触 solref | 0.02 / 1.0 |

这些是声明的仿真参数，未经真实硬件辨识。触觉来自原生接触力六通道；视觉
使用此前已接受的 RGB-D 锚点并随腕姿刚体传播，后续没有新视觉观测。释放
使用目标特定的虚拟测力桌面信号，这是明确的仿真假设。真实目标位姿和速度
仅进入评价记录器。控制日志 `state` 是本步进入前标签；native 状态才是实际
执行阶段，独立命令审计采用后者。

视频由实际全关节及物体 qpos 重建，只有运动学与灯光更新，无重新积分、插值
或物理轨迹修改。展示照明覆盖单独记入 metadata，冻结模型文件不变。

## 后续范围

冻结此完整案例作为可复现基线，停止对同一种子反复调小参数。下一阶段用同一
验收器进行独立扰动测试，再将已有工具、触觉灵巧动作和避障整合为具体多物体
任务。VLM 可随后加入目标和技能选择；VLA 未部署或训练。真人遥操作仍需人
控制虚拟机器人并保留视频和输入日志，无需真实机械臂。

历史报告 PDF、介绍视频和 ZIP 仍是此前快照；本页及本次视频不是正式提交收据。
