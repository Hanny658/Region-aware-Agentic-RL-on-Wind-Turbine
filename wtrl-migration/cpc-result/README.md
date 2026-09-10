# CPC（Region-aware Agentic CPC）— 手稿数据包

MPC 基线 + 鲁棒性）**全部可复算数据**的自足快照，供在另一台机器上克隆冻结仓库后续跑实验、写手稿。

数据源：本机 `~/wtrl/exp/*`（原始 run 目录）、`~/wtrl/*.log`（campaign 日志）、
`~/wtrl/runs/template_5mw`（被控对象与 ROSCO 配置）。逐日过程见仓库 `docs/roadmap_2026-08-30.md`
§1–§17、结论整合见 `docs/REPORT_2026-09-01.md`。

---

**环境**：`WTRL_SKIP_WIND=1 bash scripts/wsl/bootstrap.sh`（该开关跳过风库/基线生成，正是为避免
C1 重演）。OpenFAST 4.2.1 / TurbSim / torch 2.13.0 / sb3 2.9.0 / zmq 4.3.5 / openai 3.8.0 /
osqp 1.1.3；打补丁的 ROSCO 已编译且含 `zmq_client` 符号。

**跨机一致性验收**：本机以零残差重跑 U8/U12.5/U15 @ S3，对拷贝来的基线得到 DEL 降幅 -0.00 %、
能量 -0.00 %、转速比 1.000、桨距行程 51/51 与 37/37；`tq_off_s0` 的 held-out 端到端复算 F=11.7624
vs Mac 的 11.7640。历史数字与本机数字在同一口径上。

**本轮新增的实验**（全部在同一 canonical 风库上）：

| | 内容 | 结果落在 |
|---|---|---|
| A1 | `mono` x5 + `llm_fork`/`random_fork` 补 s0-s2、s5-s6（单库、n=7） | roadmap §21 |
| B1 | 单提案 LLM（`--supervisor llm --no_dry_run`）x5 | §21 |
| A3 | tower 目标臂的 TI14 / TI22 / U18 压力扫（修 C2） | §17 |
| A2 | GSPI + ROSCO 塔架阻尼器参考控制器（修 C4） | §18 |
| B2 | 600 s 计分窗重评 + MPC 的窗口/实现分解（修 C5） | §19 |
| B3 | 配对统计工具（配对 t + 精确置换 + 分辨率下限） | `scripts/dev/stats_table.py` |
| C | **第二组互斥 held-out 风场 S7-S10**，34 个 run + MPC 全部重评 | §22 |
| - | fork 决策机制分析（零算力） | §21b，`scripts/dev/fork_analysis.py` |

**一次事故与修复（2026-09-09，已闭环）**：`~/wtrl/run.sh` 曾无条件 `export WTRL_HOME`，覆盖了外层
显式设置，导致一次 `make_baselines` 把 600 s 风场的基线写进 canonical 目录（12 个 S3-S6 npz）。
12:43 之前的所有评估不受影响；文件已从本包原样恢复并逐字节校验通过。根因两处已修：run.sh 改用
`${WTRL_HOME:-$HOME/wtrl}`，`make_baselines.py` 默认跳过已存在基线（要覆盖需 `--force`）并在开头
打印风库/基线目录。详见 roadmap §20。

**仍缺**：`.env`（`LLM_BASE_URL / LLM_API_KEY / LLM_MODEL`，不在 git 里）——重跑 LLM 监督臂需要它。


---

## 1. 范围

**包含**：toy 筛选阶段的结论、OpenFAST campaign 1 / E1–E5′ / P / P5、night1+night2 监督者消融、
sched2 课程回放、R2 扭矩残差（负结果）、LPV-MPC 基线、鲁棒性 stress 扫（roadmap §1–17）。

**不包含**：范围之外的研究方向不在本包内描述。唯一带回本包口径的是**已完成的 IPC 负结果**
（仓库 roadmap §23），它只作为监督主张的边界被引用；相关原始 run 产物不在本包内。

一个例外见 **C2**：目前唯一做过 stress 扫的 RL 集合 `ipc_off_s0–4` 虽然出自 IPC campaign，但它
`ipc_max=0`，是纯 collective-pitch 的 `spec+guard`，因此属于本半段。

---

## 2. 目录结构

```
document/cpc-result/
├─ README.md                    ← 本文件（口径、警告、续跑命令）
├─ MANIFEST_external_data.csv   ← 必须单独拷贝的大文件清单（风库 + GSPI 基线）
├─ runs/<run>/                  ← 60 个 run 的产物（见 §3）
│    config.json summary.json evals.csv decisions.jsonl episodes.csv
│    eval_*.json eval_*.csv  ckpt_best.pt ckpt_last.pt  [llm_transcript.jsonl]
├─ mpc/                         ← LPV-MPC 调参网格 + held-out(S3-S6) + stress + S7-S10
├─ mpc600/  mpc600s/            ← 600 s 窗口，以及 600 s 风场上的 150 s 窗口（B2 分解）
├─ schedules/                   ← 蒸馏出的 knob 课程表（sched2 两个 arm 用的就是它）
├─ aggregate/                   ← 汇总表 + 重建脚本（纯 stdlib）
│    all_evals.csv                 每个 (run, eval tag) 一行，其它表的原料
│    table_wind_sets.csv           两组 held-out 风场上的全部臂（手稿主表）
│    table_paired_stats.csv        全部配对检验 + 精确置换下限（C6 口径）
│    table_heldout_main.csv        S3-S6 主表      table_heldout2_s78910.csv  S7-S10 主表
│    table_paper_metrics.csv       论文四指标      table_robustness.csv       压力扫
│    table_torque.csv              扭矩负结果      table_mpc.csv              MPC 全部评估
│    knob_trajectories.csv         监督者旋钮轨迹（课程图的数据）
├─ env/                         ← 被控对象与配置快照：template_5mw/（含 DISCON.IN）、
│                                  ppo.yaml reward.yaml turbine/ rosco_patch/
├─ campaigns/                   ← 实际跑这些实验的 campaign 脚本
├─ logs/                        ← 对应的 campaign 日志（含失败过程）
└─ docs_snapshot/               ← roadmap / REPORT / 文献综述 / 图（CPC 半段，§1-§22）
```

中间检查点 `ckpt_000*.pt` 未拷贝（只留 `ckpt_best` / `ckpt_last`）；如需学习曲线用
`episodes.csv` + `evals.csv`。总体积约 37 MB。

重建全部汇总表：

```bash
python3 document/cpc-result/aggregate/build_tables.py     # 无第三方依赖
```

---

## 3. run 清单（60 个 run + 3 个 MPC 目录）

| run | arm | 目标 | 说明 |
|---|---|---|---|
| `tq_off_s0-4` | **spec + guard（主线固定权重臂）** | tower | 名字有误导：这是扭矩 campaign 的对照臂，也是主线 guard 数据 |
| `n1_mono_s0-4` | **mono（单体策略对照）** | tower | 2026-09-09 重跑；与 `tq_off` 逐种子配对 |
| `n1_llmfork_s0-6` | spec + llm_fork | tower | **7 seed**；s0-2 于 2026-09-09 在本机单库重跑，s5-6 新增，s3-4 来自 Mac |
| `n1_randfork_s0-6` | spec + random_fork | tower | 同上，7 seed |
| `n1_llmsingle_s0-4` | 单提案 LLM（无分叉、无孪生试跑） | tower | B1 臂，隔离"验证机制" |
| `sched2_ep_s0-4` | 课程回放（episode 索引） | tower | 课程表见 `schedules/` |
| `sched2_comp_s0-4` | 课程回放（competence 索引） | tower | 见 C7（门控污染，未复跑） |
| `tq_on3_s0-4` | spec + guard + R2 扭矩残差 | tower | 负结果主证据 |
| `tq_on_s0`, `tq_on2k_s0`, `tq_on2kg_s0`, `tq_on2kk_s0` | 扭矩 debug 序列 | tower | 三条假路 + 真 bug 的证据链（roadmap §15） |
| `toy_dtau_{0,250,500,2000,fix500}` | toy 扭矩扫 | blade | 1-DOF 孪生 |
| `ipc_off_s0-4` | spec + guard（blade 目标） | blade | blade 目标臂；旧版压力扫用的就是它（见 C2） |
| `gspi_td` | **GSPI + ROSCO 塔架阻尼器** | tower | A2：12 组增益扫 + held-out + 压力类 + 恒等检查 |
| `_migration_check` | （基础设施） | - | GSPI 恒等检查，2026-09-09 基线恢复的凭证 |
| `mpc/` | LPV-MPC | tower | 调参网格 + held-out + 压力类 + S7-S10 |
| `mpc600/` | LPV-MPC，600 s 窗口 | tower | B2 |
| `mpc600s/` | LPV-MPC，600 s 风场上的 150 s 窗口 | tower | B2 的窗口/实现分解 |

每个 run 保留 `config.json summary.json evals.csv decisions.jsonl episodes.csv eval_*.{json,csv}
ckpt_best.pt ckpt_last.pt`（LLM 臂另有 `llm_transcript.jsonl`）；中间检查点 `ckpt_000*.pt` 未拷贝。
总体积约 37 MB。


---

## 4. 主要结果（全部由 `aggregate/*.csv` 复算：`python3 aggregate/build_tables.py`）

### 4.1 两组 held-out 风场上的全部臂（`table_wind_sets.csv`）

| arm | n | S3-S6 mean ± std | strict | S7-S10 mean ± std | strict |
|---|---|---|---|---|---|
| **llm_fork** | 7 | **17.89 ± 2.02** | 6/7 | **15.84 ± 1.63** | 7/7 |
| 单提案 LLM | 5 | 17.92 ± 1.57 | 4/5 | 15.75 ± 1.78 | 5/5 |
| schedule 回放（ep 索引） | 5 | 15.61 ± 3.68 | 5/5 | 13.83 ± 3.04 | 5/5 |
| spec + guard | 5 | 13.93 ± 2.29 | 5/5 | 12.54 ± 1.44 | 5/5 |
| mono | 5 | 13.80 ± 2.71 | **0/5** | 12.25 ± 3.10 | **4/5** |
| random_fork | 7 | 12.36 ± 5.30 | 7/7 | 10.87 ± 4.55 | 7/7 |
| guard + 扭矩残差 | 5 | 4.11 ± 2.92 | 5/5 | - | - |

所有臂在 S7-S10 上的绝对 F 都低 1.4-2.2（那组风场更难），但配对差全部保持。

### 4.2 配对检验（`table_paired_stats.csv`；精确 = 双侧符号翻转置换）

| 对比 | S3-S6 | S7-S10 |
|---|---|---|
| **llm_fork - random_fork** | +5.53，7/7 为正，精确 p = **0.0156**（n=7 下限） | **+4.97，7/7，精确 p = 0.0156** |
| llm_fork - guard | +3.48，5/5，t p = 0.041 | +2.66，5/5，t p = 0.027 |
| llm_fork - 单提案 LLM | -0.52，p = 0.76 | -0.54，p = 0.61 |
| random_fork - guard | -1.96，p = 0.56 | -2.55，p = 0.35 |
| schedule - guard | +1.69，p = 0.10 | +1.28，p = 0.18 |
| guard - mono | +0.12，p = 0.96 | +0.29，p = 0.89 |
| guard - guard+扭矩 | +9.81，5/5，t p = 0.007 | - |

**结论（相对旧版本已反转）**：决定性的是**提议者**（LLM 远好于随机搜索，两组风场各自 p = 0.016）；
**分叉验证在均值上不带来任何东西**（p = 0.76 / 0.61）；合规性由护栏 + best-ckpt 提供，与二者无关
（`random_fork` 7/7 strict 却载荷最差）。机制见 roadmap §21b：LLM 的候选**集**在验证前就好
（fork F 均值 +4.93 vs -26.2），且走出一条方向一致的课程（λ_R3 6.5x、w_speed 7.1x、Δβ_R3 0.20x），
随机搜索七个 run 走完回到 0.94-1.20x。

### 4.3 论文四指标、鲁棒性、扭矩

见 `table_paper_metrics.csv` / `table_robustness.csv` / `table_torque.csv`，以及 roadmap §17
（压力扫：塔基收益 TI8-TI22 不变、41/42 strict、U18 归零）、§18（ROSCO 阻尼器无任何 strict 配置）、
§19（600 s 窗口：RL 侧 ±1.5 F 内、tier 全不变；MPC 的转速比越过 1.0）。


---

## 5. 口径警告（2026-09-10 更新；已消解的标注为 DONE）

- **C1 DONE**｜两套风库的问题已消除：`mono`、`llm_fork` s0-2、`random_fork` s0-2 全部在本机单一
  canonical 风库上重跑，s5-6 新增。**注意这次重跑改变了结论**（见 4.2）：旧的"三个监督变体打平"
  是跨库配对的产物。旧机的 s0-2 原始数据仍然丢失，但已不再被任何结论依赖。
- **C2 DONE**｜压力扫已在 tower 目标臂上重跑（roadmap §17）；旧的 blade 目标那张表已删除。
- **C3**｜MPC 的 held-out 行必须用 `heldoutW_*` / `heldout2W_*`（按风速重打区域标签、两侧都重打）。
  `heldout_s3456_*` 是修复前的原生标签配对，**不要引用**。同理 `gspi_td` 的所有评估都用
  `--relabel_wind`（阻尼器改的是 ROSCO 自己的桨距命令）。
- **C4 DONE**｜GSPI 基线确实关掉了 ROSCO 的减载功能，但把塔架阻尼器打开并按同一准则扫参后，
  **12 组配置无一进入 strict**（roadmap §18）：它与 MPC 落在同一条"用调速换塔载"的取舍上。
- **C5 DONE**｜600 s 计分窗重评：逐 run 偏移 ±1.5 F 以内、tier 全部不变（roadmap §19）。130 s
  窗口不偏置结论。唯一例外是 MPC，其优势是窗口依赖的。
- **C6**｜统计口径（仍然适用）：n 个种子时精确符号翻转检验的下限是 2/2^n（n=5 -> 0.0625，
  n=7 -> 0.0156）。`table_paired_stats.csv` 同时给出 `exact_perm_p` 与 `exact_floor`，两者相等时
  必须写成"小于等于该值"。配对 t 与精确检验都报。
- **C7**｜`sched2_comp` 的判定仍受门控-回滚交互 bug 污染（s2: 4/5 条、s3: 1/5 条）。bug 已于
  2026-09-02 修复但**未干净复跑**；"能力索引不优于 episode 索引"应标注为受污染，或在文中从略。
- **C8**｜`ckpt_last` 经常 degraded；任何数字都取 `ckpt_best`（每个 run 都留了
  `eval_*_ckpt_last.json` 作为证据）。
- **C9**｜sched2 的课程表蒸馏自 `n1_llmfork_s3`（`schedules/`），不是 night1 的 E5 课程表（已丢失）。
  "schedule σ=0.3 最稳"的旧说法已撤回。
- **C10**｜`tq_off_*` 就是主线 guard 臂。
- **C11（新）**｜**tier 计数不是稳健统计量**。`mono` 在 S3-S6 是 0/5 strict、在 S7-S10 是 4/5——
  后者把所有控制器的转速比整体压低约 0.05，而 mono 贴在边界（1.008-1.020）。论文必须报**配对连续量**
  （mono 比 spec 多花 +0.05 转速 std，两组风场 10/10 种子，p = 0.0202 / 0.0066），tier 作为推论。
- **C12（新）**｜LLM 臂依赖外部模型 `gpt-5.6-luna`；s3/s4 跑在 Mac（2026-09-01），s0-2/s5-6 跑在本机
  （2026-09-09）。模型版本漂移无法验证，需在文中声明。


---

## 6. 大文件：风库与配对 GSPI 基线

这些文件不在 `cpc-result/` 里，而在同级的 `wtrl-migration/{wind,openfast}/`；
`MANIFEST_external_data.csv` 逐个列出 size 与 mtime，供在新机器上核对同一性。

| 类别 | 内容 | 是否已归档 |
|---|---|---|
| 风库 | `{U8,U12.5,U15}_TI8_S1-S6` 训练/监督评估/held-out | 是 |
| 风库 | `{U8,U12.5,U15}_TI8_S7-S10` **第二组 held-out（roadmap §22）** | 是（2026-09-10 补入） |
| 风库 | `{…}_TI14_S3,S4`、`U15_TI22_S3,S4`、`U18_TI8_S3,S4`、`{…}_TI2_S1,S2` 压力/课程类 | 是 |
| 基线 | `~/wtrl/baselines/openfast/<同名>.npz` 零残差 GSPI 轨迹（F 的分母） | 是 |
| 风库 | `~/wtrl/wind600`（650 s 实现，roadmap §19） | **否**，862 MB，可重新生成 |
| 基线 | `~/wtrl600/baselines/openfast`（600 s 配对基线） | **否**，462 MB，随上一条重建 |
| 基线 | `~/wtrl/baselines/toy` | **否**，45 MB，由已归档的 .bts 确定性重建 |

未归档三项的重建命令写在 manifest 的 `note` 列里。注意：重新生成 `wind600` 会得到**不同的风场实现**，
§19 的具体数字将不可精确复现（其定性结论——RL 侧对窗口不敏感、MPC 敏感——仍可复现）。

> **已归档的部分必须原样拷贝，不要在新机重新生成**：重新生成会得到不同的 .bts，届时新数字与本包内
> 的历史数字不再 seed-paired（这正是 C1 的成因，而 C1 一旦发生就会改变结论——见 §5 C1）。

---

## 7. 续跑清单（2026-09-10；A/B/C 级已全部完成）

已完成：tower 目标压力扫（C2）、GSPI+塔架阻尼器基线（C4）、600 s 重评（C5）、单库重跑与补种子
（C1）、配对统计工具（C6）、第二组 held-out 风场（评估风场特异性）。

剩余（均为可选）：

### 7.1（可选）`sched2_comp` 干净复跑（修 C7，5 run 约 7 h）

门控 bug 已在 `llm/supervisor.py` 修复；直接重跑 `campaigns/campaign_sched2.sh` 的 comp 臂即可。
若不跑，则在文中把"能力索引 vs episode 索引"标注为受污染并从略。

### 7.2（可选）监督机制消融（K=2/5 候选、fork 长度 1 vs 2 wave、去掉频谱侧信息）

roadmap §10 列过、从未做。现在"提议者重要而验证不重要"已经成立，这些消融只是补充机制证据；
§21b 的决策日志分析已经给出了主要机制解释，成本更低。

### 7.3（可选）第三组风场 / 更多种子

`llm_fork - random_fork` 已在两组独立风场上各自达到 n=7 的精确检验下限（p=0.0156）。若要更小的
p，需要 n>=9 且符号保持一致（n=9 -> 下限 0.0039），约 8 个 run 约 6 h。


---

## 8. 环境

| 组件 | 版本 |
|---|---|
| OpenFAST | v4.2.1（conda-forge） |
| ROSCO | v2.10.5 + 本仓库 `controllers/rosco_patch/`（ZMQ 22 通道、扭矩偏置、WSE 见应用扭矩） |
| Python | 3.11.16；torch 2.13.0；numpy 2.4.6；gymnasium 1.2.3；fatpack 0.7.8（rainflow/DEL） |
| 机型 | NREL 5 MW onshore；恒扭矩 `VS_ConstPower=0`；`PS_Mode=1`；控制步长 10 ms |
| 硬件 | Intel MacBook，8 训练 worker；300 episode 的 run 约 75–96 min |

`build_tables.py` 只用标准库，任何 Python ≥3.9 都能跑（用于纯写作机器）。