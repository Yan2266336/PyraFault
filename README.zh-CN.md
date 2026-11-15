*[English](README.md) · 简体中文*

# PyraFault 与 ZoomTrace

**跨倍率溯源验证（Cross-Magnification Provenance Verification，CMPV）。** 给定一张低倍率
的上下文图 `C`、一张高倍率的查询图 `Q`，以及一个**声称的**来源位置 `ℓ`，判断 `Q` 的像素是否
真的来自 `ℓ`，并恢复真实来源 `ℓ*`。

本仓库是论文对应的代码：如何获取 TCGA 切片、如何由它们构建 PyraFault 基准数据集、如何构建并
运行 ZoomTrace，以及如何让每一个 baseline 走同一套协议。

**基准数据集本身不在这里。** 已发布的五队列基准数据集——2,920 张 TCGA 诊断切片上的
3,057,732 条带坐标监督的样本——作为独立的数据集发布。本仓库提供的是完整的构建与评测代码，
外加一份**十张切片的示例**：同样的十个阶段、同样的参数，跑在约 13 GB 的原始像素上。下文所有
命令都是针对这个示例写的，可以照着原样执行。

| | |
|---|---|
| 基准数据集构建 | `scripts/00_*` … `scripts/09_*`，由 `configs/example10.yaml` 驱动 |
| ZoomTrace | `pyrafault/verify/zoomtrace.py`，经 stage 12–13 构建与运行 |
| Baseline | `pyrafault/verify/` 下各一个模块，通过 `--verifier 模块:类名` 选择 |
| 第三方提交 | [`VERIFIER_PROTOCOL.zh-CN.md`](VERIFIER_PROTOCOL.zh-CN.md)——不依赖本仓库代码的接口约定 |

**基准数据集记录的是坐标，不是像素。** 一条记录是
`(slide_id, level-0 边框, target_mpp, output_size, extraction_version)`。
`data/03_materialized/` 下的 PNG 和 `data/04_features/` 下的嵌入都是派生数据——把哪个
`rm -rf` 掉，基准数据集都不受影响。stage 09 不是声明这一点，而是**证明**它：只用坐标重新
推导 5% 的样本，再逐一比对 SHA-256。

---

# 运行

## 一次性准备

```bash
# 1. 环境。激活它，或者把 PYRAFAULT_PY 指向它的 python ——
#    本仓库不写死任何路径，用你自己的即可。
conda activate pyrafault                      # 或者：
export PYRAFAULT_PY=/path/to/your/env/bin/python

"${PYRAFAULT_PY:-python}" -c "import openslide, timm, torch; print('ok')"

# 2. 原始切片，位置由配置里的 `source:` 决定。示例用的是 `data/00_source/gdc_download`；
#    若用 `mode: directory`，则是 `slides_dir` 指的目录。十张示例切片共 12.87 GB：
python scripts/download_tcga.py          # gdc-client 的用法见「获取原始切片」
ls -l data/00_source/gdc_download        # -> 示例的 GDC 下载目录

# 3. 环境变量里的 HuggingFace token，供 UNI 特征提取器使用
[ -n "$HF_TOKEN" ] && echo "HF_TOKEN present"
```

第 1 步若失败，先重建环境——这是整条链上风险最高的依赖，值得在做任何事之前先验证：

```bash
sbatch --wait -p day -c 4 --mem=16G -t 180 --wrap \
  "conda env create -f environment.yml -p /path/to/your/env"

# 然后导出，供 SLURM 模板定位：
export PYRAFAULT_PY=/path/to/your/env/bin/python
```

## 整条流水线，一条命令

```bash
bash slurm/run_all.sh
```

换一个队列，就是换一个配置加一个独立的 data root——用环境变量传入即可，不必改动任何被
git 跟踪的文件，`data/` 里的示例也不会被碰：

```bash
PYRAFAULT_CONFIG=configs/my_cohort.yaml PYRAFAULT_DATA=data_my_cohort \
  bash slurm/run_all.sh
```

两者不设置时都回落到示例（`configs/example10.yaml`、`data/`）。**要一起换**：把新配置指向
旧的 data root，会把两个队列混进同一套表里。

这会把 stage 00 → 09 作为一条 `afterok` 依赖链提交：每个阶段只在前一个成功后才启动，所以
某一步失败会让下游全部停住，而不是让下一个阶段在半张写坏的表上继续构建。读像素的阶段
（02、03、05、08）以「每片一个任务」的数组形式运行，数组规模来自 stage 01 生成的切片列表，
因此会自动跟随队列规模。

观察进度：

```bash
squeue -u $USER
tail -f logs/pf09-qc-*.out          # 最后一个阶段——验收闸门
```

从某个阶段续跑（参数必须**补零**，写 `05` 而不是 `5`）：

```bash
bash slurm/run_all.sh 05
```

## 什么算跑成功

当 stage 09 打印出下面这些，整条链就完成了：

```
invariants: 0 violation(s)
config provenance: 0 violation(s)
leakage audit (best-threshold balanced accuracy, fit in-sample):
  border_distance: 0.527
  centre_distance: 0.530
  context_mpp: 0.500
  on_grid: 0.500
  query_mpp: 0.500
  ...
leakage: 0 violation(s)
reconstruction: 14 contexts + 50 queries
after reconstruction: 0 violation(s)
QC PASSED
```

**`QC PASSED` 且 0 violations** 是验收判据。任何一条违规都会以退出码 1 结束并中断整条链。
完整报告在 `data/05_qc/qc_report.md`。

## 单独跑一个阶段，或者单张切片

每个阶段脚本都接受 `--slides`（单片调试），以及 `--config` 和 `--data-root`。后两者默认是
`configs/example10.yaml` 和 `data/`，所以**只要不是在调试这个示例，就要显式传**——否则一次调试运行
会写进已冻结的 benchmark。**不要用 `srun`**——如果你的 shell 本身已经在一个 SLURM 作业里，
嵌套的 `srun` 会挂住。用 `sbatch --wait`：

```bash
sbatch --wait -p day -c 4 --mem=32G -t 60 -o logs/%x-%j.out --wrap \
  "PYTHONPATH=$PWD ${PYRAFAULT_PY:-python} scripts/03_build_context_candidates.py \
     --config configs/my_cohort.yaml --data-root data_my_cohort \
     --slides TCGA-55-8094-01Z-00-DX1"
```

**stage 03、05、08 分两段执行**，这一点很容易踩坑。不加 `--gather` 时，脚本只做逐片阶段：
它会把分片写到 `05_qc/stage03/`（等等）下，每张切片打印一行看起来一切正常的输出，然后
**以退出码 0 结束，却不产出该阶段的表**。于是下一个阶段因为文件缺失而报错，把人指向错误的
阶段。所有切片跑完后，gather 要作为独立的一次调用执行——`slurm/run_all.sh` 也正是把它作为
单独的 `afterok` 作业提交的：

```bash
PYTHONPATH=$PWD ${PYRAFAULT_PY:-python} scripts/03_build_context_candidates.py \
  --config configs/my_cohort.yaml --data-root data_my_cohort --gather
```

`pyrafault` 没有以 pip 方式安装，所以直接运行脚本需要 `PYTHONPATH=$PWD`。`slurm/` 下的模板
已经帮你导出了这个变量。

stage 05 需要 GPU：

```bash
sbatch --wait -p gpu --gpus=1 -c 4 --mem=48G -t 240 -o logs/%x-%j.out --wrap \
  "PYTHONPATH=$PWD ${PYRAFAULT_PY:-python} scripts/05_sample_visual_diversity.py \
     --config configs/my_cohort.yaml --data-root data_my_cohort --slides <slide_id>"
```

---

# 源数据放在哪

`configs/example10.yaml` 里的 `source.gdc_dir` 指向**一个同时包含切片和 GDC 导出文件的目录**——
`gdc-client` 下载 cart 时就是这么写的，stage 00 也是相对它来定位那三个导出文件。参考结构：

```
data/00_source/
├── gdc_download/                        # 输入 —— source.gdc_dir 指向这里
│   ├── gdc_manifest.<日期>.txt          #   三个 GDC 导出文件，在顶层
│   ├── gdc_sample_sheet.<日期>.tsv
│   ├── metadata.cart.<日期>.json
│   ├── 0c5b9279-…-5c21d73ef417/         #   每个文件一个目录，以其 GDC uuid 命名
│   │   └── TCGA-55-8094-01Z-00-DX1.<uuid>.svs
│   ├── 20303b02-…-b83bec2187dd/
│   │   └── TCGA-55-6986-01Z-00-DX1.<uuid>.svs
│   └── …
└── source_metadata/                     # 产物 —— stage 00 把三个导出文件归档到这里
```

`source_metadata/` 是流水线写的，不是你放的。不要往里面放东西。

**两种摆法，得到的结构完全相同：**

*数据放在工程目录内* —— 直接下载到位：

```bash
gdc-client download -m <manifest>.txt -d data/00_source/gdc_download
```

*数据在别处* —— 切片留在 scratch、高速盘或共享只读挂载上，用软链接到同一位置：

```bash
ln -sfn /scratch/you/gdc_download_20260812_183036 data/00_source/gdc_download
```

或者不用软链，直接在配置里写绝对路径：

```yaml
source:
  gdc_dir: /scratch/you/gdc_download_20260812_183036
```

流水线**从不写入**这个目录——它只读打开切片，并把三个小的导出文件复制一份到
`source_metadata/`。所以指向共享目录或只读挂载都是安全的。

## 如果切片不是来自 GDC

stage 01–09 完全不依赖 GDC。它们只从 `slides.parquet` 读 `slide_id`、`case_id` 和
`source_relpath` 三列；倍率、尺寸、金字塔层都来自 stage 01 用 OpenSlide 打开文件读出——
所以 **OpenSlide 能读的格式都可以**：SVS、NDPI、MRXS、TIFF、DICOM WSI。

只有 stage 00 是和数据来源相关的，而且它是可插拔的**适配器（adapter）**，不是固定不变的
基础：它唯一不可替代的职责，是产出 `slides.parquet` 的那三列，`source.mode` 决定具体怎么产
出。目前有两种适配器：

| `source.mode` | 需要 | 做什么 |
|---|---|---|
| `gdc`（默认） | `source.gdc_dir` + 三个 GDC 导出文件 | 和以前完全一样：对齐 manifest × sample sheet × cart metadata 三方，逐个核对文件大小与 MD5，任何不一致都中止整个 stage |
| `directory` | `source.slides_dir`、`source.pattern`、`source.case_id_from` | 遍历目录，找出匹配 `pattern` 的文件，从文件名推出 `slide_id`，按配置的规则解析 `case_id`，并记录一份自己算出来的 sha256 |

`mode` 缺省时默认为 `gdc`，所以 `configs/example10.yaml` 不需要改动，它的 digest 也不受影响——
`configs/example_directory.yaml` 是一份完整、可直接使用的 `directory` 模式配置示例，每个字
段都带注释。

`slide_id` 取文件名去掉最后一级扩展名，它是此后所有阶段、文件名和记录使用的身份标识。
这一规则是推导出来的、不可配置，所以首次运行前值得先看一眼你的文件名。注意两个适配器不必给出
相同结果：把示例自己的 GDC 下载目录交给 `directory` 模式，`slide_id` 会是
`TCGA-44-2655-01Z-00-DX1.ee255271-...`（整个主干，含 UUID），而 `gdc` 模式会剥掉 UUID。
两者都唯一且稳定，只是同一张切片的不同命名——因此经由不同适配器构建的 benchmark 无法逐切片
比对。每个队列选定一个适配器后就不要中途更换。

这个差异比听上去要窄，值得说清楚。把示例的十张切片用 `directory` 模式重跑，context 层是
**逐项完全一致**的——同样 187 个 context、同样 429 个候选、同样的 case→split 分配，因为
context 的选取只依赖组织掩膜和配置的几何参数。差异出现在 query 层（20,169 个候选 vs 冻结运行
的 20,066）：为了让格点不可反推，query 网格按 `sha256(seed | context_id | 轴)` 做了抖动
（见「已知限制」），而 `context_id` 里嵌了 `slide_id`。所以适配器改变的是抽到哪些 query 位置，
而不是某张切片能产出多少 context、或落进哪个 split。

对 `mode: directory` 而言，最需要留心的字段是 `case_id_from`，因为划分是 **case（病人）
级**的：同一病人的所有切片必须落在同一个 split 里，`case_id` 填错会让同一个病人悄悄同时出
现在 `dev` 和 `test` 中，评估结果偏乐观却没有任何机制能发现。三种写法：

- `case_id_from: "slide_id"` —— 一个病人一张切片；`case_id` 直接复制 `slide_id`。本仓库试
  点队列里 `case_id` 是 TCGA 病人条码（`TCGA-44-2655`），也正是 `slide_id` 的前三段，所以在
  试点队列上 `case_id = slide_id` **并不**成立——但只要你的队列确实是一个病人一张切片，这个
  写法就是对的。
- `case_id_from: "^(TCGA-[^-]+-[^-]+)"` —— 一个恰好带一个捕获组的正则表达式，作用于
  `slide_id`，捕获组的内容就是 case。某个 `slide_id` 匹配不上时会直接报错并点名该切片，绝
  不会静默退回到用 `slide_id` 本身。
- `case_id_from: "case_map.csv"` —— 指向一个带 `slide_id,case_id` 两列的 CSV 文件的路径。
  发现的切片如果在映射表里找不到，会报错并点名该切片。

**关于完整性校验，如实说明。** `directory` 模式没有外部的真相来源可供核对——没有发布方给
出的 manifest。它记录的 `sha256_local` 和 `file_size` 是直接从文件本身算出来的，建立的是一
个**基线**，用于日后检测文件是否发生变化（重跑 stage 00 再比对），而不是像 `gdc` 模式的 MD5
核对那样，对照发布方的记录做校验。不要把两者当作同等强度的保证。

用法：把 `configs/example_directory.yaml`（或它的副本）指向你的目录，然后照常从 stage 00
开始跑（`bash slurm/run_all.sh`，或者 `PYTHONPATH=$PWD python
scripts/00_ingest_source.py --config configs/example_directory.yaml`）。

---

# 在新队列上运行

这套流水线与具体队列无关——试点用的十张切片是**数据，不是结构**。用一批新的 GDC 下载构建
基准数据集的步骤：

**1. 指向新的下载目录。** 重指符号链接，或修改 `source.gdc_dir`：

```bash
ln -sfn /path/to/gdc_download_YYYYMMDD_HHMMSS data/00_source/gdc_download
```

`configs/example10.yaml` 里那三个 GDC 导出文件名是通配符
（`gdc_manifest.*.txt`、`gdc_sample_sheet.*.tsv`、`metadata.cart.*.json`），所以新的 cart
下载不需要改这里。匹配到 0 个或多个都会报错并列出找到了什么——**绝不会静默取第一个**。

**2. 设置划分规模**，在 `configs/example10.yaml` 里。两种写法都可以：

```yaml
splits:
  dev: 6            # 绝对例数——三者之和必须等于队列的 case 总数
  calibration: 2
  test: 2
```
```yaml
splits:
  dev: 0.6          # 或者比例，和为 1.0——队列规模还不确定时更保险
  calibration: 0.2
  test: 0.2
```

**3. 复核最小 context 门槛。**

```yaml
splits:
  min_contexts_per_slide: {test: 15, calibration: 15}
```

这条门槛把几何上过于贫瘠的切片挡在评估集之外。15 这个值适配的是「最大切片能产出约 200 个
合格 context」的队列；标本更小的队列可能需要调低。如果没有任何划分方案满足它，stage 03 会
失败并打印每片的 context 数——那是**调低门槛的信号，不是重试的信号**。

**4. 跑起来。**

```bash
bash slurm/run_all.sh
```

其余部分会自动适配：level-0 网格按**每张切片自己的 MPP** 逐片计算（试点里就已经混着 0.2520
和 0.2527 两种像素尺寸，对应 14×14 和 15×15 两种网格），没有 MPP 的切片会被排除而不是猜测，
SLURM 数组规模跟随切片列表。

> **加入新切片会让整个基准数据集重新冻结。** context 选择、query 选择和 pair 构造都是全局
> 进行的，所以旧的 `pair_id` 不再指向同一条记录。请把新队列当作**一次新的冻结**来规划，而不
> 是一次增量扩充。

---

# 你会得到什么

基准数据集本体在 `data/01_benchmark/` 下——**不到 1 MB**，因为它存的是坐标：

| 文件 | 内容 |
|---|---|
| `pairs.parquet` | 任务记录：`(C, Q, ℓ, ℓ*)` 加标签与溯源信息 |
| `queries.parquet` | 每个 query 的真实来源边框 `ℓ*` 及其抽取规格 |
| `contexts.parquet` | 每个 context 的边框、组织占比与描述子 |
| `splits.parquet` | case 级的划分归属 |
| `slides.parquet` | 源切片，含金字塔与 MPP 元数据；`source.mode: gdc` 下经 MD5 校验，`directory` 下记录 sha256 基线 |
| `*_candidates.parquet` | 完整的合格候选池——选择与挖掘过程的审计轨迹 |
| `config.yaml` | 产出这次冻结所用的**确切**配置 |
| `benchmark_manifest.json` | 上述每张表的 sha256 与行数 |

`data/` 下的其余内容都是派生的、可删除的：`02_cache/`、`03_materialized/`（PNG）、
`04_features/`（嵌入）、`05_qc/`（掩膜、overlay、报告）。

`pairs.parquet` 是任务表——一条记录一行，共 39 列。真正会用到的是这些：

| 列 | 含义 |
|---|---|
| `pair_id`、`slide_id`、`context_id`、`query_id` | 身份，以及这条记录由什么构建 |
| `claimed_bbox_l0_{x,y,w,h}` | `ℓ`，`slide_id` 的 level-0 像素 |
| `true_bbox_l0_{x,y,w,h}` | `ℓ*`，**`true_slide_id`** 的 level-0 像素 |
| `true_slide_id`、`same_slide` | 返回的 patch 来自哪张切片；除跨片两类外均等于 `slide_id` |
| `degradation_digest`（在 `target` 里，不是表的列） | 返回的 `query` 被施加的退化；未退化时为 `""`。harness 用它做定位缓存的 key：退化按 item 抽取，所以两个 item 可以共享切片、context 和 `true_bbox`，却分别被喂了 5° 和 15° 的旋转 |
| `claim_correct` | 标签——只有 `clean` 为 `True` |
| `fault_type`、`fault_severity` | 八类中的哪一类，以及偏移档位（`small`/`medium`/`large`，其余为空） |
| `normalized_offset`、`offset_frac_context`、`iou_claim_true` | 声称位置离真值多远；前两者在跨片时为空，因为那时距离没有意义 |
| `true_within_context`、`local_recovery_possible` | `ℓ*` 是否在 `C` 内——四类替换样本为 false |
| `hardness_score` | 冻结表征空间里 query↔claim 的余弦，用于按难度分层 |
| `context_mpp`、`query_mpp`、`scale_ratio` | 倍率组合，使多份冻结版本可以拼接 |
| `degradation_band` | 这条记录属于哪个成像退化档位：`none` / `mild` / `moderate` / `severe` |
| `degradation_kind`、`degradation_params` | 在返回的 `Q` 上重放的操作（`none` 或 `geometry`）及其冻结参数，JSON 形式 |
| `served_pixel_sha256` | 实际返回的 `Q` 的 RGB 缓冲区 sha256，在 QC 抽到该行之后写入（在那之前是 `""`） |
| `split` | `dev` / `calibration` / `test`，按 **case（病人）** 分配 |

**成像退化这一维。** 在 `C` 中定位 `Q` 不应只靠灰度模板匹配就能解决，因此**返回的** query patch
带有一个按行冻结的几何退化：绕 patch 中心旋转，外加一个很小的尺度误差，双线性采样，边界处做钳位，
所以不会有填充色进入画面。`degradation_band` 给出档位——`none`（0°）、`mild`（5°）、
`moderate`（10°）、`severe`（15°）——`degradation_params` 则记录该行抽到的 `angle_deg` 与 `scale`。

在此之上开发前，有四点需要知道：

- **参数是冻结的，不会在读取时重抽。** 加载器从行里读出参数并重放，所以这些像素在另一台机器上
  可以逐字节复现；`pyrafault/degrade.py` 里没有任何一步交给实现可能变化的库滤波器或编码器。
  `served_pixel_sha256` 就是对这一点的校验：阶段 09 第一次抽到某行时把它冻结，之后的运行再做比对，
  于是一台会生成不同像素的机器会**直接报错**，而不是悄悄给出基于不同图像算出的指标。
- **`none` 档就是恒等变换。** 像素、缓存键、哈希都与一份从不知道退化这回事的构建完全一致。本仓库
  `configs/example10.yaml` 给每一行都分配 `none`，所以示例的像素没有任何改变；而携带四个档位的冻结版本会在每个
  `(split, fault_type)` 单元内均衡分配档位，因此档位无法用来预测 `claim_correct`。
- **档位不在 `inputs` 里。** 验证器不会被告知它的 `Q` 是否被旋转过、旋转了多少；像素是唯一的证据。
  它也不在 `target` 里：打分方从 `pairs.parquet` 读取它。
- **按档位分别汇报。** `score_predictions` 会返回 `by_degradation_band`
  （[`VERIFIER_PROTOCOL.zh-CN.md`](VERIFIER_PROTOCOL.zh-CN.md) 第 8 节）；对一份多档位的冻结版本
  只给一个汇总数字，等于把它专门要衡量的那一维藏了起来。

其余几张表是审计轨迹而非任务本身：`contexts` 与 `queries` 记录每个区域自身的抽取规格与质量
度量，`*_candidates` 是选取时所依据的完整候选池，`slides` 是每张切片的金字塔与 MPP 元数据，
`splits` 是 case 到 split 的分配。用 `pyrafault.schema.read_table(path, name)` 读取任意一张，
它会先校验列结构再返回。

一份冻结版本的规模由配置里的三个数字和切片自身的几何条件决定：每张切片
`context.num_per_slide` 个 context（上限 30，组织放不下那么多的切片会少于这个数）、每个
context `query.num_per_context` 条 query，以及每条 query 八对——`clean` 加七种 fault 各一。
因此十张切片的示例产出的是几千对，而不是三百万条；具体数字由 stage 09 打印出来。已发布的
基准数据集的规模见论文，这个示例既不复现它，也不声称复现它。

`configs/example10.yaml` 里八类**全部开启**，所以由它构建出的冻结版本带有完整的标签词表。其中四类属于
「替换」——实际服务的 patch 并非声称位置所描述的那一块：

| `fault_type` | 错在哪 | `ℓ*` 怎么选 |
|---|---|---|
| `out_of_context` | patch 根本来自切片的其他地方 | 在不与 `C` 接触的候选中均匀随机 |
| `out_of_context_hard` | ……而且来自看起来最像所请求内容的组织 | 与 query 自身冻结表征余弦最近的候选 |
| `out_of_slide` | patch 来自**另一张切片**——缓存/并发导致的失败 | 在同一 split 的其他切片中均匀随机 |
| `out_of_slide_hard` | ……而且来自该 split 内看起来最像的组织 | 跨其他切片的余弦最近候选 |

`out_of_slide` 由 `faults.out_of_slide.enabled: true` 单独启用。它的替换来源**只从同一
split 的切片中取**：把 dev 切片的像素拉进 test 样本，和把切片本身同时放进两个 split 是一回事，
没有哪个 fault 类型有权撤销 case 级切分。QC 会逐行断言这一点。

这两类的 `true_bbox_l0` 索引的是**另一张切片的坐标系**，所以记录里带 `true_slide_id`
（其他类型下等于 `slide_id`），加载器按它读取 query 图像。`iou_claim_true` 恒为 0、
`normalized_offset` 恒为空——两个不同切片坐标系之间的距离不是一个"小数值"，而是一个无意义的
数值，置空可以防止它被平均进任何统计里。

**跨片替换是最容易的一类，不是最难的。** 在一份十张切片的 10×→40× 冻结版上实测，query↔claim 余弦中位数依次为
`out_of_slide` 0.155 < `out_of_context` 0.333 < `random_same_context` 0.461 <
`out_of_slide_hard` 0.465 < `hard_mismatch` = `out_of_context_hard` 0.741。它还可能带有
同片故障所没有的染色与扫描仪线索。它衡量的是「最基本的 provenance 失败能否被发现」；
**不要不加说明地把它的准确率与同片类型合并汇报**。

两者的**声称位置都留在 `C` 内**——那是 Agent 实际请求的位置——落在外面的是真实来源，所以
`true_within_context` 与 `local_recovery_possible` 均为 false。验证器无法靠包含性检查否掉
它们，只能去判断「相似的组织」不等于「同一处组织」。两类同时启用会给每条 query 各加一个条件；
stage 09 会按条件分别报告 query↔claim 余弦中位数，让挖掘效果可以被核对而不是被假定：全片搜索
应当让 `out_of_context_hard` 高于 `hard_mismatch`，后者高于 `random_same_context`，再高于
`out_of_context`。划分为 case 级 6 dev / 2 calibration / 2 test。特征来自 UNI
（`MahmoodLab/UNI`，revision `b55a5ec6cade1a39edfe6534189a9b8ca7a022f0`，1024 维）。

---

# 使用基准数据集

```python
from pyrafault.dataset import PyraFaultDataset

ds = PyraFaultDataset("data/01_benchmark", "data/00_source/gdc_download",
                      split="test", materialize=True)

item = ds[ds.item_ids[0]]

item["inputs"]    # context 图、query 图、claimed_bbox、context_bbox、
                  # context_mpp、query_mpp、item_id、slide_id、split …
item["target"]    # claim_correct、true_bbox、true_slide_id、fault_type、pair_id …
                  # 仅用于打分
```

`materialize=True` 会命中缓存的 PNG；不加则按冻结的坐标从原始切片现读。两条路径产出的字节
完全相同。

## 验证器会收到什么

每条样本就是下面这些——没有别的可用，也没有哪一项是可选的：

| 字段 | 类型 | 含义 |
|---|---|---|
| `context` | `PIL.Image`，RGB，`context.size_px`²（发布版为 2048²） | 低倍上下文图 `C` |
| `query` | `PIL.Image`，RGB，`query.size_px`²（发布版为 512²） | 高倍 patch `Q` |
| `context_bbox` | `BBoxL0(x, y, w, h)` | `C` 在切片上的位置，单位是 **level-0 像素** |
| `claimed_bbox` | `BBoxL0(x, y, w, h)` | 声称来源 `ℓ`，level-0，恒在 `context_bbox` 内 |
| `context_mpp` | `float`，µm/px | `mc` |
| `query_mpp` | `float`，µm/px | `mq`；`mc / mq` 就是把 `C` 对齐到 `Q` 所需的尺度系数 |
| `slide_id` | `str` | `C` 所在的切片 |
| `item_id` | `str`，sha256 十六进制 | 不透明句柄，不含任何标签信息 |
| `split` | `str` | `dev` / `calibration` / `test` |
| `extraction_version` | `str` | 像素是怎么产生的 |

四个 bbox 数值都是 `slide_id` 的 level-0 像素，所以 `claimed_bbox` 与 `context_bbox` 可以直接
比较；把一个 level-0 框换算成 `context` 图像内的位置是
`(box.x - context_bbox.x) / context_bbox.w * context.width`。

`target` 是答案，产出判断的过程中不得读取：`claim_correct`（bool）、`true_bbox`（`ℓ*`）、
`true_slide_id`（`true_bbox` 所在的切片——只有两类跨切片 fault 才会不同于 `slide_id`）、
`fault_type`、`fault_severity`、`normalized_offset`、`iou_claim_true`、`hardness_score`、
`pair_id`。

`degradation_band` 两边都不在：验证器拿到的像素本身就是它们被退化过的唯一证据，而按档位拆分结果
是持有 `pairs.parquet` 的打分方的事。

## 验证器必须返回什么

每个 `item_id` 一条记录。这是所有 baseline 和 ZoomTrace 共同被打分的契约，返回别的形式就无法
互相比较：

```python
{
    "item_id":        str,            # 原样回传
    "estimated_bbox": BBoxL0 | None,  # ℓ̂ —— 视觉上最可能的来源，slide_id 的 level-0
                                      # 像素。只有在方法完全放弃定位时才为 None。
    "reliability":    float,          # s —— 越高表示「这次定位越有判别性」。任意单调
                                      # 标度即可；它只被阈值化，不跨方法直接比较。
    "decision":       str,            # "verified" | "mismatched" | "unverifiable"
}
```

这只是方法的 `localise()` 返回的 `Localisation` 的概念形态——不是你要提交打分的文件。
完整的落盘提交格式（`predictions.parquet`，全部十一个 `PREDICTION_COLUMNS`）见
[`VERIFIER_PROTOCOL.md`](VERIFIER_PROTOCOL.md) 第 5 节（中文版见
[`VERIFIER_PROTOCOL.zh-CN.md`](VERIFIER_PROTOCOL.zh-CN.md)）。

`decision` 由前两项对照可靠性阈值 `τ` 与空间容差 `δ` 得出：`reliability < τ` 时为
**`unverifiable`**；否则 `ℓ̂` 与声称位置在 `δ` 内一致则为 **`verified`**，不一致则为
**`mismatched`**。`unverifiable` 是方法的**拒识**行为，不是第三种真值类别，数据集中没有任何
一条记录带这个标签。

各指标分别读取哪个字段：

| 指标 | 读取 |
|---|---|
| AUROC / AUPRC | `reliability` 连同 `estimated_bbox` 与 `claimed_bbox` 的一致程度——用连续分数，不用离散化后的 `decision` |
| balanced accuracy、指定误验率下的 TPR | `decision` |
| 归一化中心误差、IoU、Recall@k | `estimated_bbox`，仅对 `true_bbox` 落在 `context_bbox` 内的样本 |
| risk–coverage、ECE、Brier | `reliability`，以 `unverifiable` 作为弃权 |
| source recovery accuracy | `mismatched` 样本上的 `estimated_bbox` |

有两条规则决定结果是否可比，且它们都是**评测框架**的性质而非数据的性质：

1. **逐条打分。** 每条记录的输出只是它自己那份 `(context, query, claimed_bbox, …)` 的函数。
   为吞吐而做 batching 没有问题——在 `eval()` 且无跨样本运算时，N 个一批与分别跑 N 次在算术上
   完全一致——但如果打分规则读取了同一 context 内的其他样本，就会重新打开「已知限制」里那条
   格点残留。
2. **`τ` 和 `δ` 在 `calibration` 上确定，在 `test` 上冻结。** 它们是分数分布上的阈值，在 `test`
   上估计等于用评测集去调评测。`dev` 用于开发，`calibration` 用于定这两个数，`test` 用于产出
   你要报告的数字。

`inputs` 里带有 `context_mpp` 与 `query_mpp`，也就是任务定义里的 `mc` 和 `mq`。跨倍率方法
需要它们：把 `C` 对齐到 `Q` 的物理分辨率就是对二者比值做算术，没有它们，方法的第一步就无法进行。
它们在一次冻结内是常数、因而不含标签信息，但泄露审计仍按数值给它们打分而不是豁免——一旦把多个
倍率组合合并进同一张表，而各组合的 fault 构成又不同，倍率就成了不看像素的标签预测器。注意流水线
用的是 µm/px 而非「×」：这些切片标称 40×，实测却是 0.2520–0.2527 µm/px，所以 0.25 的目标需要
上采样，配置校验器会直接拒绝。

每条 pair 还记录 `context_mpp`、`query_mpp` 与 `scale_ratio`，这样不同倍率下构建的冻结版本可以
拼成一张表做分层。比值单独不能作键：5×→20× 与 10×→40× 的比值都是 4.0，只在绝对分辨率上不同。

**`inputs`/`target` 的拆分是一条安全性质，不是排版。** 仅 `true_bbox` 一项就回答了恢复任务，
而 `normalized_offset == 0.0`、`iou_claim_true == 1.0`、非空的 `fault_severity`、以及
`hardness_score` 的空/非空模式，每一个都在复述标签。它们全部位于 `target`，所以要拿到答案
必须**刻意伸手去取**。

`inputs` 携带的是不透明的 `item_id = sha256(seed|pair_id)` 而非 `pair_id`，因为 `pair_id`
的构造是 `f"{query_id}__{fault_type}"`——`"__clean" in pair_id` 就能以 100% 精确率恢复
`claim_correct`。`ds.item_ids` 与 `ds[item_id]` 覆盖遍历和索引；`ds.pair_ids` 与
`ds[pair_id]` 依然可用，供确实需要可读标识符的工具使用（例如按 `pairs.parquet` 或 QC
overlay 文件名做关联）。

**评估路径上没有任何随机性**——没有随机裁剪、没有颜色增强、不重跑组织过滤、不重选 query。
两个独立构造的数据集实例对同一条记录返回逐字节相同的图像，并有测试断言这一点。

## 运行验证器

stage 12 构建验证器要匹配的检索窗口库（需要 GPU）；stage 13 跑一个验证器并写出预测；
stage 14 打分：

```bash
python scripts/12_build_window_bank.py --config configs/example10.yaml --data-root data

# calibration：先拟合 tau 和 delta，再对该 split 自身做出判定并写出预测
python scripts/13_run_verifier.py --config configs/example10.yaml --data-root data \
    --split calibration --fit-thresholds --matcher sift --audit-independence

# test：用 calibration 刚冻结的阈值做判定
python scripts/13_run_verifier.py --config configs/example10.yaml --data-root data \
    --split test --matcher sift

python scripts/14_score_verifier.py \
    --predictions data/06_verify/zoomtrace_sift/predictions_test.parquet \
    --data-root data --split test
```

`scripts/13_run_verifier.py` 启动时就调用 `Thresholds.load()`，所以在任何 split 被判定之
前——包括拟合 `τ`、`δ` 所用的 `calibration` 本身——阈值必须已经存在。`--fit-thresholds`
就是用来打破这个死结的：在传入的那个 split 上，它先对每条样本只做定位、不做判定，从这些
定位结果里拟合出两个阈值（`pyrafault/verify/thresholds.py` 里的 `fit_tau`、`fit_delta`），
写出 `thresholds.json`，然后才对同一个 split 做判定并写出它自己的预测文件。这个参数只在
`calibration` 上传一次；之后的 split（如 `test`）不带这个参数，直接加载这一步写出的
`thresholds.json`。`--matcher sift` 是纯 CPU 的，stage 13 本身不需要申请 GPU；
`--matcher loftr` 是基于学习的替代方案。

baseline 走同样的三条命令,只需把 ZoomTrace 的默认值换成 `--verifier 模块:类名`——不需要
窗口库、不需要 matcher,输出在 `06_verify/<name>/` 下。不写任何代码就想看这条路径跑通,
用随仓库提供的骨架 `--verifier pyrafault.verify.ncc_baseline:NccBaseline`。

这里复现的完整协议——每一个 `PREDICTION_COLUMNS` 字段、判定规则、完全不进入本仓库也能
提交预测文件——见 [`VERIFIER_PROTOCOL.md`](VERIFIER_PROTOCOL.md)（中文版
[`VERIFIER_PROTOCOL.zh-CN.md`](VERIFIER_PROTOCOL.zh-CN.md)）。


## ZoomTrace 与全部 baseline

ZoomTrace 是 `--verifier zoomtrace`（默认值），也是唯一需要 stage 12 窗口库的方法。每个
baseline 是 `pyrafault/verify/` 下的一个模块，用 `--verifier 模块:类名` 选择，输出写到
`data/06_verify/<它自己的名字>/`。对所有方法，三条命令都是同一套：在 `calibration` 上拟合、
对 `test` 做判定、打分。

| 方法 | `--verifier` | 外部资产 |
|---|---|---|
| **ZoomTrace** | `zoomtrace` | 窗口库（stage 12）；`--matcher sift` 或 `loftr` |
| Claim-conditioned LoFTR | `pyrafault.verify.plugins:ClaimConditionedPlugin` | 无 |
| Efficient LoFTR | `pyrafault.verify.plugins:EfficientLoFTRPlugin` | `EFFICIENT_LOFTR_SOURCE`、`EFFICIENT_LOFTR_CHECKPOINT` |
| SuperPoint + LightGlue | `pyrafault.verify.plugins:LightGluePlugin` | `LIGHTGLUE_SOURCE`、`SUPERPOINT_CHECKPOINT`、`LIGHTGLUE_CHECKPOINT` |
| MINIMA–LightGlue | `pyrafault.verify.plugins:MinimaLightGluePlugin` | `LIGHTGLUE_SOURCE`、`SUPERPOINT_CHECKPOINT`、`MINIMA_LIGHTGLUE_CHECKPOINT` |
| MatchAnything–ELoFTR | `pyrafault.verify.plugins:MatchAnythingELoFTRPlugin` | `MATCHANYTHING_ELOFTR_SOURCE`、`MATCHANYTHING_ELOFTR_CHECKPOINT` |
| RoMa | `pyrafault.verify.plugins:RoMaPlugin` | `ROMA_SOURCE`、`ROMA_CHECKPOINT`、`ROMA_DINOV2_WEIGHTS` |
| SUE | `pyrafault.verify.plugins:SUEPlugin` | `PYRAFAULT_DINOV2_SNAPSHOT` |
| Statistical Uncertainty | `pyrafault.verify.plugins:StatisticalUncertaintyPlugin` | `PYRAFAULT_DINOV2_SNAPSHOT` |
| NCC | `pyrafault.verify.ncc_baseline:NccBaseline` | 无 |

每个插件都从环境变量读自己的路径，并且在构造时不加载任何模型，所以即使是重型 baseline，
`--from-localisations` 也是零成本的。`python scripts/check_baseline_assets.py` 会报告哪些
变量已设置、对应的文件或目录是否存在——在提交作业之前，而不是失败之后。上游的固定 revision
与权重哈希记录在各模块的 `revision` 字串里，并写进每份预测文件，所以一次运行会指明产出它的
那组权重。

`PYRAFAULT_BASELINE_DEVICE` 取 `auto`、`cpu` 或 `cuda`，对所有插件生效。

claim-conditioned baseline 是式 (1) 独立性唯一被允许的例外：它读 `inputs["claimed_bbox"]`、
声明 `reads_claim = True`，而框架的回应是把 claim 并入它的去重键，并对它拒绝
`--audit-independence`。为什么两者都是必需的，见 `pyrafault/verify/base.py` 里的
`Verifier.reads_claim`。

视觉语言模型 baseline 是 `pyrafault/verify/vlm_baseline.py` 里的 `VlmBaseline` 与
`VlmLocaliser`，API 版本是 `vlm_api.py` 里的 `AnthropicVlmLocaliser`。它们需要一个模型
标识符而非零参数构造，所以是作为库类调用而不是走 `--verifier`；API 版本从环境变量读取
`ANTHROPIC_API_KEY`。

### 大 split 的分片跑法

定位与判定是可分离的，这正是大 split 跑得起来的原因：昂贵的那一半作为数组作业跑，便宜的那一半
读它写下的结果。

```bash
# 每个数组任务一个分片；每个只读自己那些 context 的窗口库
python scripts/13_run_verifier.py --config configs/example10.yaml --data-root data \
    --split test --localise-only --shard "$SLURM_ARRAY_TASK_ID" --n-shards 40

# 然后一次性读入所有分片做判定；不读像素，不需要 GPU
python scripts/13_run_verifier.py --config configs/example10.yaml --data-root data \
    --split test --from-localisations
```

分片是按 context 分组的，不是按条目跨步切分，所以一次运行内部的去重在分片之后依然成立。
判定那一步遇到缺口的分片集合会直接拒绝，而不是对一个子集打分。

---

# 各阶段速查

| 阶段 | 产出 | 是否读像素 |
|---|---|---|
| 00 接入源数据 | `slides.parquet` —— 由 `source.mode` 选择适配器；`gdc` 核对大小与 MD5，`directory` 遍历目录 | 否 |
| 01 扫描切片 | MPP、物镜倍率、金字塔层；排除无 MPP 的切片 | 仅文件头 |
| 02 组织掩膜 | `05_qc/masks/*.npz` — 积分图，`tissue_fraction` 查询为 O(1) | 缩略图 |
| 03 冻结 context | `context_candidates.parquet`、**`contexts.parquet`**、`splits.parquet` | 仅选中的 context |
| 04 query 候选 | `query_candidates.parquet` — context 内网格 | **否** |
| 05 多样性采样 | **`queries.parquet`** — 聚类分层，每个 context 5 个 | 每个候选 |
| 06 构造 pair | `clean` / `adjacent_drift` / `random_same_context` | 否 |
| 07 挖掘难负例 + 替换类 | `hard_mismatch`（父 context 内余弦 top-1），以及四类替换样本：`out_of_context{,_hard}`、`out_of_slide{,_hard}` | 否 |
| 08 物化 | PNG 缓存（派生）+ 回写 `image_sha256` | context + query |
| 09 质检 | 不变量、重建校验、泄露审计、overlay、`qc_report.md` | 5% 抽样 |

stage 04 **完全不读像素**——它只查积分图，这也是它以单个快速任务运行而非数组的原因。

stage 03、05、08 是先逐片、再 gather：逐片阶段写分片，另一次带 `--gather` 的调用才把分片
汇成该阶段的表。`slurm/run_all.sh` 会把 gather 作为独立的 `afterok` 作业提交。

---

# 目录结构

| 路径 | 内容 |
|---|---|
| `pyrafault/` | 库代码——被单元测试覆盖的部分 |
| `scripts/NN_*.py` | 每个阶段一个 CLI，是库之上的薄封装 |
| `configs/` | `example10.yaml`（十张切片的示例；用到它的那次运行会把它逐字复制进 `data/01_benchmark/`）与 `example_directory.yaml`（一份可用的非 GDC 配置） |
| `slurm/` | 每阶段一个 sbatch 模板，外加 `env.sh` 与 `run_all.sh` |
| `tests/` | 1,611 个测试；其中标记 `-m slow` 的 35 个需要真实切片和计算节点 |
| `data/` | 由运行产生，除 `01_benchmark/benchmark_manifest.json` 外均被 gitignore；另一个队列放在 `data_*/`，同样被忽略 |
| `examples/` | 标识那十张示例切片的三个 GDC 导出文件 |

---

# 环境

`environment.yml` 只钉了 Python 版本，其余全部浮动。基准数据集的嵌入依赖于产出它们的那套
**确切**软件栈，所以那次求解被单独冻结在 `environment.lock.yml` 里。

| 文件 | 用途 |
|---|---|
| `environment.yml` | 宽松规格；新建环境的起点 |
| `environment.lock.yml` | 产出已发布嵌入的确切环境 |

lock 文件钉住了 `torch==2.13.0`、`timm==1.0.28`、`transformers==5.15.0`、
`openslide-python==1.4.6`、`openslide-bin==4.0.1.2` 等。要逐位复现 stage 05 的输出，需要的是
这一套确切版本，而不是重新对 `environment.yml` 求解一次。导出命令记录在 lock 文件的头部作为
溯源信息——**不要重新执行它**，否则会用当前环境覆盖掉那份冻结记录。

## 特征提取器

`configs/example10.yaml` 使用 `uni`（`MahmoodLab/UNI`），这是一个受控仓库，需要持有 `HF_TOKEN`
的账号在模型页面上接受一次授权协议。若未接受，stage 05 会以一条指明模型页面的错误失败。
无需授权的后备方案：

```bash
PYRAFAULT_EXTRACTOR=dinov2 bash slurm/run_all.sh 05
```

事后切换只需重跑 stage 05 和 07。提取器名称、解析出的 HF commit 和嵌入维度都逐行记录，且
manifest 在 gather 阶段由各切片分片**一次性**写入、分片之间必须一致——所以混用提取器的表会
大声失败，而不会被意外产出。

---

# 测试

```bash
PY="${PYRAFAULT_PY:-python}"

$PY -m pytest -m "not slow"                      # 591 个测试，几分钟，不需要切片

sbatch --wait -p day -c 4 --mem=32G -t 40 -o logs/%x-%j.out --wrap \
  "PYTHONPATH=$PWD $PY -m pytest -m slow"        # 10 个测试，真实切片，计算节点
```

慢测集承载着最关键的几条保证：跨进程逐字节一致的读取、仅凭坐标的重建、以及两个独立构造的
加载器实例返回相同图像。

---

# 已知限制

**查询格点曾经可被推导出来——已通过按 context 抖动修复。**
`random_same_context` 和 `hard_mismatch` 把 `ℓ` 放在查询网格的格点上，`clean` 把它放在
`ℓ*`（同样是格点）上；而 `adjacent_drift` 以一段连续距离偏移 `ℓ`，必然落在格点之外。当网格
原点**居中**时，这个原点可以仅凭 context 边框和切片 MPP 推算出来——而这两者是任何可用的评测
接口都必须暴露的——于是一次取模运算就能标出 935 对，占整个基准数据集的 25%，且**一个像素都
不用看**。更糟的是它标出的每一对都是 fault，等于白送给验证者四分之一样本上正确的
`claim_correct = False`，直接抬高总体准确率。

修复方式是给每个 context 的网格原点施加一个确定性的、两轴独立的抖动，使得格点归属**无法从
任何单条 `(C, Q, ℓ)` 判定**。修复后实测：同一条规则的精确率是 0.750，恰好等于数据集的 fault
基准率——提升为零。

**对于按 context 汇集样本的验证器仍有残留**（而非逐条打分）。在当前设计下不可消除：`clean`
按定义就在格点上，而 `adjacent_drift` 必须离开格点才能达到最小的偏移严重度带，所以在同一个
context 内部，被偏移的那些样本构成少数派余数类。

**预期的协议是逐条打分，在该协议下这条残留是关闭的。** 每条记录仅由它自己的 `(C, Q, ℓ)` 打分；
取众数那套检验需要同一 context 的多条样本，只有一条时无从计算。注意分组键是可见的——`inputs`
里带 `context_bbox`，一个 context 贡献的约 40 对全都共享它，因为一个定位不到 `C` 在切片何处的
验证器根本无法完成任务。所以关闭这条残留的是**协议而不是数据**，评测方案里应当写明它是逐条打分。

张量 batching 不等于汇集：在 `model.eval()` 且没有跨样本运算时，N 个样本一批与分别跑 N 次在
算术上完全一致，而这里用到的提取器全部是冻结的。真正会破坏它的是「打分规则读取同组其他样本」，
或者阈值在 `test` 上重新估计而不是从 `calibration` 冻结下来。

**`hard_same_slide.pool: slide` 被拒绝，而这个拒绝本身就是结论。** 把 `hard_mismatch` 的
搜索范围放宽到全片，会让**声称位置**落到 `C` 之外，而 `C` 之外的声称位置一次 bbox 包含性检查
就能否掉、一个像素都不用看——最难的那类样本会变成最容易的。全片范围的难负例是反过来构造的，
即下面的 `out_of_context_hard`：声称位置不动，被替换的是**返回的 patch**。设成 `pool: slide`
会直接失败并说明这一点。

stage 09 的泄露审计（`pyrafault.qc.report.check_leakage`）在每次冻结时重新测量上述全部内容，
任何不看像素的信号一旦超过 0.55 的平衡准确率就让构建失败。它覆盖 on-grid 检验、几何距离、
声称框的尺寸，以及加载器暴露的**每一个字符串字段**——最后一项的由来是：早期版本的加载器曾
通过 `pair_id` 泄露标签（其后缀就是 fault 类型），而只测几何量的审计看不见它。

---

# 获取原始切片

那十张全切片图像**不在**本仓库中——它们合计 12.87 GB。`examples/gdc_metadata/` 里放着标识
它们的三个 GDC 导出文件，你可以据此取到**完全相同**的那批切片：manifest 里记录了每个文件的
UUID、发布方给出的大小和 MD5。

这十张是已发布队列中最先通过 stage 01 分辨率与可读性检查的十张，一个 case 一张，覆盖该队列
全部五个 project：TCGA-BLCA、TCGA-LGG、TCGA-LUAD、TCGA-LUSC、TCGA-UCEC。全部是开放获取的
诊断切片，因此不涉及任何 GDC token。

两个工具都能填好下载目录。`gdc-client` 是官方工具，对大批量下载更合适，因为它支持断点续传和
并行：

```bash
# https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
gdc-client download -m examples/gdc_metadata/gdc_manifest.2026-08-12.143046.txt \
                    -d data/00_source/gdc_download
cp examples/gdc_metadata/* data/00_source/gdc_download/
```

`scripts/download_tcga.py` 除本仓库的依赖之外什么都不需要，它会顺带把那三个导出文件放到位，
并跳过已经按发布大小存在的文件：

```bash
python scripts/download_tcga.py                      # 全部十张，12.87 GB
python scripts/download_tcga.py --limit 2            # 只取前两张，试一下这条路径
python scripts/download_tcga.py --dry-run            # 只列出将要下载什么
```

两者写出的都是 `<file-uuid>/<file-name>.svs`——这是 `gdc-client` 的产出布局，也是
`slides.parquet` 里 `source_relpath` 记录的路径。平铺式的下载会让这里构建出的冻结版本无法与
官方工具构建出的互相校验。

在 `source.mode: gdc` 下，stage 00 会在任何其他步骤之前逐个核对文件的大小与 MD5——所以被截断
或被替换的下载会让流水线停住，而不是悄悄改变基准数据集。（`directory` 模式没有可比对的发布方
记录，只记录自己算出的 sha256 基线，见上文「如果切片不是来自 GDC」。）

`data/` 下的任何东西都不在本仓库中。已发布的基准数据集表格作为独立数据集发布；这里的
`data/01_benchmark/` 是你自己那次运行的产出，而 `benchmark_manifest.json` 是让第二次运行
能够与第一次核对的 sha256 与行数记录。

# `examples/` 里有什么

| 路径 | 内容 |
|---|---|
| `gdc_metadata/gdc_manifest.*.txt` | 十张切片的文件 UUID、发布名、MD5 与大小 |
| `gdc_metadata/gdc_sample_sheet.*.tsv` | 按文件 UUID 关联的 project、case、sample 与标本字段 |
| `gdc_metadata/metadata.cart.*.json` | 每个文件的 case UUID、实验策略与访问级别 |
| `gdc_metadata/README.md` | 十张切片各自的 project、case、体积与 MPP |

stage 00 会按文件 UUID 对这三份做内连接，并且拒绝丢行——所以某个文件只出现在其中一份里时，
运行会指名停住，而不是让它悄悄从队列中消失。

---

# 这里没有什么

| | 原因 |
|---|---|
| 基准数据集表格 | 作为独立数据集发布；本仓库负责构建它们 |
| Agent 工具链 harness | 论文第 7 节的实验，不属于基准数据集或验证器本身 |
| 画图与制表脚本 | 它们读取本仓库不发布的结果文件 |
| 设计规范 | 由作者保管 |

`pyrafault/verify/` 下的每一个验证器都在这里，包括表现不好的那些；第三方不跑这里任何代码就能
提交预测所需的协议也在这里。

---

# 许可

代码按 MIT 许可发布，见 [LICENSE](LICENSE)。TCGA 切片本身不随本仓库发布，其使用受 GDC 自己的
条款约束。
