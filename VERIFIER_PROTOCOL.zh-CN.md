# 验证器协议

## 1. 目的

无论是 ZoomTrace 本身，还是之后的每一个 baseline 方法，都通过同一套 harness 运行和打分：
`pyrafault/verify/base.py`、`pyrafault/verify/runner.py`、`pyrafault/verify/metrics.py`
与 `pyrafault/verify/thresholds.py`。正是这一点让不同方法的数字可以互相比较——运行和打分
这两步都不是按方法定制的，因此报告出来的指标差异只反映定位能力的差异，而不是测量方式的差异。
本文档就是这四个模块所实现的契约，写给一个从未打开过这个仓库的作者：只要按此契约构造出一份
合规的 `predictions.parquet`，得到的数字就与 ZoomTrace 自己的数字含义相同。若本文档与代码有
出入，以代码为准——但那样的出入本身就是本文档的一个 bug，应当被当作 bug 报告出来。

## 2. 上手

下面所有内容都以「你能加载 benchmark」为前提。这一节是从 clone 到跑通一个自己的
verifier 的最短路径。

**环境。** `environment.lock.yml` 是产出本项目结果的精确解;`environment.yml` 是用来
重新求解的宽松规格。

```bash
conda env create -f environment.lock.yml -p /path/to/your/env
export PYRAFAULT_PY=/path/to/your/env/bin/python
$PYRAFAULT_PY -c "import openslide, timm, torch; print('ok')"
```

你自己的方法可能只需要 NumPy——§11 的骨架就是。

**加载样本。** `PyraFaultDataset` 负责把冻结的坐标表变成 §4 描述的 `inputs`。它需要
benchmark 目录,以及配置里 `source:` 指向的根目录——因为图像是按冻结坐标从切片现读的:

```python
from pyrafault.dataset import PyraFaultDataset

with PyraFaultDataset("data/01_benchmark", "data/00_source/gdc_download",
                      split="test", materialize=True) as ds:
    item = ds[ds.item_ids[0]]
    inputs = item["inputs"]     # 你的 localise() 收到的就是它
    target = item["target"]     # 答案，做判断时不得读取
```

`materialize=True` 在有缓存 PNG 时用缓存、否则读切片,两条路径产出的字节完全相同。
`ds.item_ids` 是整个 split,顺序稳定。

**在一个 split 上跑你的 verifier。** `localise` 在单条上跑通之后,先在 `calibration` 上
拟合阈值、冻结,再用它们判定 `test`(§9):

```python
from pyrafault.verify.runner import fit_thresholds, run_verifier

with PyraFaultDataset("data/01_benchmark", "data/00_source/gdc_download",
                      split="calibration", materialize=True) as cal:
    thresholds = fit_thresholds(MyVerifier(), cal, "calibration")
thresholds.save("thresholds.json")

with PyraFaultDataset("data/01_benchmark", "data/00_source/gdc_download",
                      split="test", materialize=True) as ds:
    predictions = run_verifier(MyVerifier(), ds, thresholds, audit_independence=True)
predictions.to_parquet("predictions_test.parquet", index=False)
```

或者用命令行。`scripts/13_run_verifier.py` 接受任何可 import 的 `Verifier` 子类,写成
`--verifier 模块:类名`,无参数实例化,然后走与 ZoomTrace 完全相同的拟合、判定、写出代码:

```bash
V=my_package.my_verifier:MyVerifier          # 必须可 import(在 PYTHONPATH 上或已安装)

python scripts/13_run_verifier.py --config configs/example10.yaml --data-root data \
    --verifier $V --split calibration --fit-thresholds --audit-independence
python scripts/13_run_verifier.py --config configs/example10.yaml --data-root data \
    --verifier $V --split test
python scripts/14_score_verifier.py --data-root data --split test \
    --predictions data/06_verify/<verifier name>/predictions_test.parquet
```

输出写到 `06_verify/<name>/`,以你的类的 `name` 命名。需要配置的 verifier 自己读取配置——
harness 不传任何方法特有的参数,这样每个方法的命令行都一样。想在写任何代码之前先看整条
路径跑通,用 `V=pyrafault.verify.ncc_baseline:NccBaseline`:就是 §11 的骨架,已作为模块
随仓库提供。**阈值是每个方法各自的**:拟合你自己的,绝不要复用别人的。

**提交前自查两件事:**

- **至少跑一次 `audit_independence=True`。** 它默认关闭,而这是唯一能机械检验你的
  `localise` 没有读取 claim 的手段(§3)。
- **先用 `scripts/14_score_verifier.py` 给自己打一次分**(§10)。提交的样本集与 split
  不匹配时打分器会报错,自己先发现比交上去再发现好。

**如果你的方法需要按 context 预计算。** ZoomTrace 为每个 context 缓存了一层窗口描述子,
由 `scripts/12_build_window_bank.py` 构建、存在 `06_verify/window_bank/` 下。协议对此
既不要求也不禁止——`localise` 是逐条调用的,可以查阅任何预计算结果,**只要那份预计算不
依赖 claim**。你若也要建一份,请放在 `06_verify/window_bank/` 之外,那个目录属于 ZoomTrace。

**一份参考实现。** `pyrafault/verify/zoomtrace.py` 是个完整的 verifier,约 140 行。想看
这些接口在真实方法里怎么用,而不只是骨架,读它。

## 3. 你要实现什么

一个 `Verifier` 子类（`pyrafault/verify/base.py`），只有一个抽象方法：

```python
def localise(self, inputs: Mapping) -> Localisation:
    ...
```

`localise` 永远不会收到 claim。`inputs` 里有一个字段叫 `claimed_bbox`，实现不得读取
它：任务把估计值 `ℓ̂` 定义为 context `C` 与 query `Q` 的函数，与 claim `ℓ` 无关。一个
偷看 `claimed_bbox` 的方法不是在验证这个 claim——它只是在复述它，这样做完全不需要真正
完成任务就能得到满分。

这一点不只是写在文档里——运行时的 harness *可以*对其进行审计，但仅当被要求这样做时。
`run_verifier` 的 `audit_independence` 标志默认为 `False`（见
`pyrafault/verify/runner.py`），而 `scripts/13_run_verifier.py` 只有在带
`--audit-independence` 调用时才会把它打开——不带这个标志，运行时不会检查这一点。标志
打开时，`run_verifier` 先计算一次 `localise(inputs)`，然后对同一个 `(C, Q)` 在一个被
扰动过的 `claimed_bbox`（偏移量很大、与数据无关）下重新计算一次，并断言两次的
`estimated_bbox` 和 `reliability` 一致。若 verifier 自己声明 `reads_claim = True`，审计是被**拒绝**而不是被判失败——这是唯一被允许的例外，用于那个「以 claim 为条件」的 baseline，它存在的意义就是量出独立定位到底买到了什么；声明它同时会把 claim 放进 harness 的去重 key，否则两个只有 claim 不同的 item 会共用一个答案。提交到 benchmark 的方法不设这个标志。（只比较这两个字段：`diagnostics` 不参与比较，
两次调用之间可以不同。）缓存命中时它也会重复这项检查，用产生这次命中的另一个真实
claim 来做——这样在一个 split 中重复出现的 `(C, Q)` 也会用一个真实的第二个 claim 而不
只是人为构造的那个来审计。开着审计运行、且在 `localise` 里读取了 `claimed_bbox` 的
验证器都会在这条断言上失败，而不是悄悄蒙混过去。一个从未运行过这个 harness 的提交
者——在本仓库之外构造 `predictions.parquet`（见第 9 节）正是这条路径——完全不会被
审计；不要指望被抓到。上一段所述的独立性要求无论是否被检查都对提交者具有约束力。

## 4. 你会收到什么

`inputs` 恰好携带以下十个字段——没有其他内容，也没有一个是可选的。这与 README 里
「What a verifier receives」一节完全一致；两者若有分歧，那是其中之一的 bug。

| 字段 | 类型 | 含义 |
|---|---|---|
| `context` | `PIL.Image`，RGB，`context.size_px`² 大小（出厂 2048²） | 低倍 context `C` |
| `query` | `PIL.Image`，RGB，`query.size_px`² 大小（出厂 512²） | 高倍 patch `Q` |
| `context_bbox` | `BBoxL0(x, y, w, h)` | `C` 在切片上的位置，level-0 像素坐标 |
| `claimed_bbox` | `BBoxL0(x, y, w, h)` | 声称的来源 `ℓ`，level-0，始终在 `context_bbox` 内——`localise` 不得读取（见第 2 节） |
| `context_mpp` | `float`，µm/px | `mc` |
| `query_mpp` | `float`，µm/px | `mq`；`mc / mq` 是把 `C` 对齐到 `Q` 所需的缩放因子 |
| `slide_id` | `str` | `C` 所在的切片 |
| `item_id` | `str`，sha256 十六进制 | 不透明句柄；不携带任何标签信息 |
| `split` | `str` | `dev` / `calibration` / `test` |
| `extraction_version` | `str` | 像素是如何生成的 |

四个 bbox 数值都是 `slide_id` 的 level-0 像素，因此 `claimed_bbox` 与 `context_bbox`
可以直接比较；一个 level-0 box 转换为 `context` 图像内部坐标的公式是
`(box.x - context_bbox.x) / context_bbox.w * context.width`。

**`query` 可能是被退化过的，而且不会告诉你是否如此。** 返回的 `Q` 带有一个按行冻结的成像退化
——绕中心旋转（`none` 0°、`mild` 5°、`moderate` 10°、`severe` 15°）外加一个很小的尺度误差
——目的是让「在 `C` 中定位 `Q`」不能只靠灰度模板匹配解决。档位记在 `pairs.degradation_band`
里，它既不在 `inputs` 也不在 `target` 中：像素是唯一的证据，这是刻意的设计。参数冻结在表里、
读取时重放，不会重抽，因此像素可以逐字节复现；`none` 档是恒等变换，与一份从不知道退化这回事的
构建完全一致（本仓库发布的三份冻结版本全部是 `none`）。结果由打分方按档位拆分——见第 8 节的
`by_degradation_band`。

## 5. 你要返回什么

`localise` 返回一个 `Localisation`（`pyrafault/verify/base.py`）：

```python
Localisation(
    estimated_bbox: BBoxL0 | None,   # ℓ̂ —— slide_id 的 level-0 像素；只有方法完全
                                      # 拒绝定位时才为 None
    reliability: float,              # 与该方法专属拟合出的 tau 做阈值比较
    diagnostics: Mapping[str, Any] = {},
)
```

`reliability` 可以使用任意单调尺度。它会与在 `calibration` 上按方法拟合出的 `τ` 做
阈值比较（见第 6、8 节），并且永远不会在不同方法**之间**比较——只会和该方法自己的 `τ`
比较。一个未归一化的互相关峰值和一个针对零假设分布的经验尾概率，作为这个值都同样合法；
没有要求它必须有界、必须校准成概率、也没有要求不同验证器之间可比。重要的只是：在同一个
方法内部，`reliability` 越高意味着这个估计越可信。

## 6. harness 会产出什么

`run_verifier` 把每个 item 的 `Localisation` 转换成
`pyrafault/verify/runner.py` 的 `PREDICTION_COLUMNS` 里的一行：

| 列 | 类型 |
|---|---|
| `item_id` | `str` |
| `estimated_bbox_l0_x` | `int`，可为空（`estimated_bbox is None` 时为 `NaN`/`None`；落盘后一旦某一行为空，整列会被上转型为 `float64`，读取方要按 `float64` 列而不是字面的 `int` 列来处理） |
| `estimated_bbox_l0_y` | `int`，可为空 |
| `estimated_bbox_l0_w` | `int`，可为空 |
| `estimated_bbox_l0_h` | `int`，可为空 |
| `reliability` | `float` |
| `decision` | `str`，取值之一：`"verified"`、`"mismatched"`、`"unverifiable"` |
| `verifier_name` | `str` |
| `verifier_revision` | `str` |
| `thresholds_digest` | `str`，sha256 十六进制——标识产生这个 `decision` 用的是哪一组 `(τ, δ)` |
| `runtime_ms` | `float`——计算出这个答案所花的成本；同一个 `(context, query)` 缓存命中时会原样复制，所以按不同的 `(slide_id, context_bbox, true_slide_id, true_bbox, degradation_digest)` key 求和得到的是总计算量，而不是逐行的墙钟时间。`degradation_digest` 是这个 key 的一部分，不是可选项：成像差异是按条目抽的，所以两个条目可以共享切片、context 和 `true_bbox`，却分别被喂了 5° 和 15° 的图。不含它的 harness 会把第一条的定位结果直接给第二条——没有任何异常，每条都有结果，也不报错。它放在 `target` 里正是为此，和同样为了防碰撞的 `true_slide_id` 并列。 |

一份合规的提交就是一份恰好包含以上这些列的 parquet 文件，被打分的 split 里的每个
`item_id` 对应一行。

## 6b. 全量规模怎么跑

单进程跑不完一个完整 freeze。ZoomTrace 每个不同的 `(C, Q)` 约 240 ms，NCC 对照约
360 ms，所以肺部 freeze 的 143,000 条 calibration 和 155,080 条 test，单方法就是 20 到
30 小时——还没算读像素，而且每个方法各自标定阈值，要跑两遍。超过多数队列的墙钟上限。

但这个任务本身没有顺序依赖：`localise` 只看一对 `(C, Q)`，而判定是
`(定位结果, 声称位置, τ, δ)` 的纯函数。stage 13 就在这条缝上拆成两半：

```bash
# 贵的那一半，每个分片一个 array job
sbatch --array=0-19%20 ... scripts/13_run_verifier.py --data-root data_lung \
    --split calibration --localise-only --shard "$SLURM_ARRAY_TASK_ID" --n-shards 20
# 便宜的那一半，跑一次，读盘上所有分片
python scripts/13_run_verifier.py --data-root data_lung --split calibration \
    --from-localisations --fit-thresholds
```

`--localise-only` 写出 `LOCALISATION_COLUMNS`——也就是 `PREDICTION_COLUMNS` 去掉
`decision` 和 `thresholds_digest`（此刻它们还不可能存在）——落到
`06_verify/<method>/localisations_<split>/<shard>of<n>.parquet`。`--from-localisations`
读入全部分片，按需标定，判定，写出与不分片跑法完全相同的 `predictions_<split>.parquet`。
这个"完全相同"是一条测试，不是一句承诺（`tests/test_run_verifier_sharded_cli.py`）。

如果你要自己重新实现这套分片，有三点值得知道：

- **按 context 分，不按 item 分。** 共享同一张被服务 patch 的 item 必然共享 context，
  所以按 `context_id` 分组能让 harness 的 `(C, Q)` 去重在跨分片后依然成立——并且让一个
  ZoomTrace 分片只需读自己那些 context 的 window bank，而不是全部 12.8 GB。
- **分组先排序再跨步取**，所以第 *k* 个分片在 array 的每个作业里、以及重跑时，都指同一批
  item。
- **缺口是拒绝，不是补齐。** 判定时只要有一个 item 没有定位结果，就带着那个 item 的 id
  退出，而不是写出一张只覆盖部分 split 的预测表；两个分片有重叠也同样拒绝。这里真正要防的
  失败形态是：每一行单独看都对，但整体只评了一部分。

`--audit-independence` 属于 `--localise-only` 那一半——`localise` 真正发生的地方；传给判定
那一半会被拒绝，而不是被悄悄跳过。

## 7. 判定规则

`DECISIONS`（`pyrafault/verify/base.py`）的三个取值是 `"verified"`、`"mismatched"`
与 `"unverifiable"`。`unverifiable` 是方法主动弃权——它不是第三个真值类别，基准数据里
没有任何一条记录会以它作为标签。

`decide(loc, claimed_bbox, query_side, tau, delta)` 依次执行：

```python
if loc.reliability < tau:
    return "unverifiable"
if loc.estimated_bbox is None:
    return "mismatched"
d = math.hypot(loc.estimated_bbox.cx - claimed_bbox.cx,
               loc.estimated_bbox.cy - claimed_bbox.cy)
return "verified" if d <= delta * query_side else "mismatched"
```

也就是说：`reliability` 低于阈值 `τ`，弃权。达到或超过 `τ`，却完全给不出估计值，
这是一次自信的拒绝——判为 `mismatched` 而不是弃权，因为此时 `reliability` 已经在
`τ` 之上了。达到或超过 `τ` 且给出了估计值时，比较 `ℓ̂` 与 `ℓ` 的中心距离与
`δ · query_side` 的大小：在容差内是 `verified`，超出是 `mismatched`。`δ` 以
query side 为单位表示，因为它就是这样被拟合出来的（见第 8 节）——作为**归一化**定位
残差的一个分位数，也就是原始像素残差除以 `true_bbox.w`。

## 8. 打分

`score_predictions(predictions, pairs, item_ids=None)`（`pyrafault/verify/metrics.py`）
返回的 dict 恰好包含以下这些 key——`metrics.json` 里没有别的内容：

| key | 内容 |
|---|---|
| `n_items` | 打分的行数 |
| `coverage` | `decision != "unverifiable"` 的行所占比例 |
| `decisions` | 每种 `decision` 取值的计数 |
| `balanced_accuracy` | 在有判定结果的行上，`decision == "verified"` 与 `claim_correct` 的对照；若这些行的 `claim_correct` 全部相同则为 `None` |
| `verification_score` | `{"mean_clean": …, "mean_fault": …, "auroc": …, "auprc": …}`——下面的 `verification_score(ρ, ℓ̂, ℓ)` 按 `claim_correct` 分组求均值，外加同一批逐条分数相对 `claim_correct` 的 AUROC、AUPRC（只有一个类别时为 `None`） |
| `localisation` | `{"n", "median_normalised_centre_error", "median_iou"}`，只在 `true_bbox` 落在 `context_bbox` 内的样本上计算 |
| `by_fault_type` | 每个 `fault_type` 的 `n`、`coverage`、`verified_rate` |
| `by_scale_ratio` | 每个 (`scale_ratio`、`context_mpp`) 组合的 `n`、`verified_rate`——两者都要，不能只用 `scale_ratio`：5×→20× 与 10×→40× 的比例同为 4.0 但 `context_mpp` 不同，只按比例分组会把两层本该分开的分层混到一起 |
| `by_degradation_band` | 每个 `degradation_band` 的 `n`、`coverage`、`verified_rate`、`balanced_accuracy`（某档位只含一个类别时为 `None`）——即成像退化这一维（见第 4 节）。`pairs` 表早于该列的冻结版本这里是空 `{}`；本仓库发布的三份全部是恒等变换，因此只有 `{"none": …}`。方法自己无法汇报这一项：档位既不在 `inputs` 也不在 `target` 中，只有持有 `pairs` 的打分方才能这样拆分结果——这也正是对一份多档位冻结版本只给一个汇总数字会把它专门要衡量的那一维藏起来的原因 |

`verification_score` 是这些均值所基于的逐条量：

```
verification_score(ρ, ℓ̂, ℓ) = ρ · IoU(ℓ̂, ℓ)      (ℓ̂ 为 None 时为 0.0)
```

单靠 `ρ` 是不够的：只要定位是有区分度的——集中、自信、低熵——`ρ` 就会很高，而这与
估计值是否与 claim 一致毫无关系。一个方法完全可能对一个错误答案表现得很自信。乘上
`IoU(ℓ̂, ℓ)` 让这个分数同时依赖两个都应该重要的量：方法有多确信，以及它是否与所声称
的内容一致。这个乘积无需额外参数，并且对两个因子都是单调的。`score_predictions` 就在
这个连续量上计算 AUROC 与 AUPRC：AUROC 用秩和（Mann-Whitney U）恒等式，AUPRC 用逐步
precision-recall 求和，两者都只用 numpy 实现，都是相对 `claim_correct` 计算的。当打分
集合只有一个类别时，两者都返回 `None` 而不是一个数——AUROC、AUPRC 在那种情况下是未定义
的，不是零。风险–覆盖（risk–coverage）曲线不计算；上表列出的内容之外没有别的。

弃权（`decision == "unverifiable"`）会被排除在准确率类指标之外——`balanced_accuracy`
只在有判定结果的 item 上计算——但无论是否弃权，每个 item 都会计入 `coverage`（有非弃权
判定的 item 占比）。弃权既不是一个错误答案，也不是一个正确答案；它是方法选择不作答的
一个陈述，必须以这种方式体现出来，而不是从分母里被悄悄剔除。

## 9. 两条规则

有两条属于 harness（而非数据）的性质，是不同运行之间、不同方法之间的结果可以互相比较
的原因。

1. **逐条打分。** 每一行的输出都只是它自身 `(context, query, claimed_bbox, …)` 的
   函数。为了吞吐量做张量 batching 是可以的：在 `eval()` 模式下，只要模型内部没有跨样本
   运算，N 条数据打成一批和分别做 N 次单条前向传播在算术上完全一致——run harness 本身
   会按 `(context, query)` key 做缓存和去重，但绝不会让某个 item 的分数依赖另一个 item。
   不可以的是**汇集（pooling）**：一条打分规则去读取共享同一个 context 的其他 item。
   `context_bbox` 在 `inputs` 里是可见的——一个 context 贡献的约 40 对样本都共享它——一条
   规则如果去看某个 context 内部「偏移样本相对干净样本」的分布，就能不读一个像素地还原出
   `claim_correct`。逐条打分正是关闭这条残留的方式；汇集会重新打开它。
2. **在 `calibration` 上拟合 `τ` 和 `δ`，在 `test` 上冻结使用。** `fit_tau` 和
   `fit_delta`（`pyrafault/verify/thresholds.py`）都只接受 `calibration` 的分数作为
   输入。若在 `test` 上估计任一阈值，就是用评测集来调评测本身——报告出来的数字就不再
   衡量泛化能力。`dev` 用于开发，`calibration` 专门用来拟合这两个数字，`test` 才是你
   报告的那个数字所来自的 split。

## 10. 不依赖本仓库提交结果

要被打分并不需要本仓库的代码，只需要遵守它的接口。为你运行过的那个 split 写一份
`predictions.parquet`，列名恰好是 `PREDICTION_COLUMNS`（见第 5 节），每个 `item_id`
一行，然后用以下命令打分：

```bash
python scripts/14_score_verifier.py \
    --predictions path/to/predictions_test.parquet \
    --data-root data \
    --split test
```

`--split` 很关键。预测结果是按 split 分开产出的（`predictions_<split>.parquet`），
而归档的 `pairs.parquet` 覆盖所有 split。`score_predictions(predictions, pairs,
item_ids=None)` 会从 `pairs["item_id"]`（若给了 `item_ids` 则以它为准）计算出期望
的 `item_id` 集合，并在提交的集合与期望集合不一致时**抛出异常**——是 `ValueError`，
不是警告——无论哪个方向的不一致都会失败：`predictions` 里有而不该有的 item，或者应该
有却缺失的 item，都会导致调用失败。这是特意设计的：如果合并操作悄悄丢掉了缺失的
item，`coverage` 仍会报告 `1.0`，`balanced_accuracy` 看起来也很合理，因为每个指标都
只在匹配上的那些行上计算。一份不完整的提交这样就会被读成一次完整的运行，而不是一次
出了问题的运行。所以——在构造你的预测文件之前——先确定你要打分的是哪个 split，并传入
`--split`（或者自己预先把 `--pairs` 过滤到那个 split）；用一份按 split 划分的预测文件
去对完整、未过滤的 `pairs.parquet` 打分会直接抛出异常，而不会悄悄地只打分一个子集。

`scripts/14_score_verifier.py` 不读取任何验证器代码——只读 `predictions.parquet` 和
基准数据的 `pairs.parquet`——所以只要你的预测文件符合第 5 节的契约，整条路径完全可以
在本仓库之外完成。

当你确实想只对一个 split 里的一部分 item 打分时——比如调试运行，或者只重新提交之前
失败的那些 item——`item_ids` 就是被认可的做法：直接调用
`score_predictions(predictions, pairs, item_ids=my_subset)`，把你打算打分的那个确切
集合传进去，这样「期望的集合」就是这个子集（而不是 `pairs["item_id"]`），上面那个
异常检查的也是你的提交是否匹配这个子集，而不是整个 split。`scripts/14_score_verifier.py`
本身没有暴露 `item_ids` 参数——从命令行只能用 `--split` 缩小期望集合，或者让 `--pairs`
指向一个已经过滤好的表；用其他方式（比如手工编辑 `pairs.parquet`）来打分一个刻意选定
的子集，不是被认可的做法。

## 11. 一个可运行的骨架

一个完整、可直接运行的归一化互相关（NCC）baseline——没有学习到的特征，除了 NumPy 和
Pillow 之外没有别的外部依赖。它用 `context_mpp`/`query_mpp` 把 `Q` 下采样到与 `C`
相同的物理分辨率，在**原分辨率**的 `C` 上**逐像素**计算下采样后 `Q` 的 NCC（用 FFT 实现，
2048² 的 context 不到一秒），并把相关峰值作为 `reliability` 报告出来。

它在仓库里的位置是 `pyrafault/verify/ncc_baseline.py`，类名 `NccBaseline`，
`--verifier pyrafault.verify.ncc_baseline:NccBaseline` 可以原样跑起来。**写你自己的方法时，
照着这个文件起步，不要照 `ZoomTrace`。** ZoomTrace 的构造函数要 window bank、特征提取器和
匹配器三个由 stage 13 替它拼好的外部对象，而且它的 `name` 是保留的（§2）；以它为模板，等于
一开始就陷进和你的方法无关的装配代码里。这个骨架构造时不接收任何参数，不需要 GPU，不需要任何
预先建好的产物，一共 69 行。

`localise` 里有四处值得原样抄，因为每一处写错都不会抛任何异常：

- **用 `context_mpp / query_mpp` 把 `Q` 下采样。** 不做这一步，两张图尺度不一致，后面全在
  量错的东西。
- **没有答案时返回 `Localisation(None, 0.0)`**——不要猜一个框，也不要抛异常。空位置是协议里
  的合法答案，判定规则把它读成有把握的否定（§7）。
- **把峰值换算回切片的 level-0 坐标系**，用 `context_bbox` 和 `ctx_bbox.w / context.width`。
  你算出的峰值是 context 图像内部的像素偏移，而你要返回的是切片坐标系里的框。这一步写错的
  表现是「定位到处都错但置信度很高」，很难联想到是坐标问题。
- **`reliability` 必须随置信度单调递增。** 判定规则假定如此。NCC 取值在 `[-1, 1]`，所以最后
  一行做了重映射；如果你的分数是越小越好（比如某种距离），得你自己翻过来。

是逐像素，不是按步长：这个骨架的第一版以模板边长四分之一为步长滑动，在十张切片的冻结版上几乎
什么都没定位到（test 上中位 IoU 0.0）。坐标本身没错——抽查的样本在真实源位置的 NCC 是
0.78–0.95——但 query 的原点被抖动到任意像素，而在组织上离开源位置 16 px 相关性就掉到 0
附近，粗网格根本采不到峰值。以此为起点的任何模板匹配 baseline 都受同样的约束。

```python
"""A minimal verifier: normalised cross-correlation of a downsampled Q over C."""
from __future__ import annotations

import numpy as np
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier


def _gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float64)


def _ncc_map(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """NCC of `template` at every valid offset of `image`, stride 1.

    Stride 1 is not an optimisation detail. A true source sits at an arbitrary pixel
    offset, and on tissue the correlation falls to about zero a few pixels away from
    it -- a coarser stride usually never samples the peak at all.
    """
    th, tw = template.shape
    t = template - template.mean()
    # Sum of (window * zero-mean template) at every offset, by FFT convolution with
    # the flipped template.
    k = np.zeros_like(image)
    k[:th, :tw] = t[::-1, ::-1]
    full = np.real(np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(k)))
    cross = full[th - 1:, tw - 1:]
    # Per-window variance from integral images.
    def window_sum(a):
        s = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return s[th:, tw:] - s[:-th, tw:] - s[th:, :-tw] + s[:-th, :-tw]
    n = th * tw
    s1 = window_sum(image)
    var = np.maximum(window_sum(image ** 2) - s1 ** 2 / n, 0.0)
    denom = np.sqrt(var * (t ** 2).sum())
    return np.divide(cross, denom, out=np.zeros_like(cross), where=denom > 1e-6)


class NccBaseline(Verifier):
    name = "ncc_baseline"
    revision = "2"

    def localise(self, inputs) -> Localisation:
        context, query = inputs["context"], inputs["query"]
        scale = inputs["context_mpp"] / inputs["query_mpp"]  # mc / mq
        side = max(1, round(query.size[0] / scale))
        c_arr = _gray(context)
        t_arr = _gray(query.resize((side, side), Image.LANCZOS))

        if c_arr.shape[0] < side or c_arr.shape[1] < side:
            return Localisation(estimated_bbox=None, reliability=0.0)

        scores = _ncc_map(c_arr, t_arr)
        y, x = np.unravel_index(int(np.argmax(scores)), scores.shape)
        best = float(scores[y, x])

        ctx_bbox = inputs["context_bbox"]
        px_scale = ctx_bbox.w / context.width
        est = BBoxL0(
            int(round(ctx_bbox.x + x * px_scale)),
            int(round(ctx_bbox.y + y * px_scale)),
            int(round(side * px_scale)),
            int(round(side * px_scale)),
        )
        # NCC is in [-1, 1]; rescale to [0, 1] so `reliability` behaves monotonically
        # with the same sign convention the decision rule assumes (higher is better).
        return Localisation(estimated_bbox=est, reliability=(best + 1.0) / 2.0)
```

这段代码原样随仓库提供为 `pyrafault/verify/ncc_baseline.py`——页面与模块一旦不一致,
`tests/test_ncc_baseline.py` 就会失败;该测试还会把这个模块依次跑过 `fit_thresholds`、
开启审计的 `run_verifier` 和 `score_predictions`。

这只是一个 baseline，不是一个有竞争力的方法：单一灰度模板、不做旋转或尺度搜索，在难样本
上会输给任何基于特征匹配的方法。它存在的意义是展示一个 `Verifier` 所需的最小形态——
读取 `inputs`、绝不读取 `claimed_bbox`、返回一个 `Localisation`——让作者在构建自己
真正的方法之前，先有一个能跑通全流程的东西。

## 12. 需要注意的一点

**`τ` 约束的是什么。** *误验*（false verification）指一个错误的 claim 被**接受**——该样本
越过了 `τ`，并且它的估计在 `δ` 内与 claim 一致。一个越过 `τ` 但估计与 claim 不一致的错误
样本会被判为 `mismatched`，那是正确答案、是成功。因此 `fit_tau` 除可靠性外还接收一致性判定，
而 `δ` 先于 `τ` 拟合，以保证拟合 `τ` 时一致性已有定义。

**目标值是 0.01，而且它属于 benchmark，不属于你的方法。** `fit_tau` 把误验率压在
`pyrafault.verify.thresholds.TARGET_FVR` 之下。你要拟合自己的 `τ`，但必须对着这个数拟合：
对着不同目标拟合出的两个 `τ`，描述的弃权行为是不可比的——而让各方法可比，正是把拟合放在
共享代码里的全部意义。

之所以是 0.01，是因为 0.05 被实测为形同虚设。在退化版肺切片冻结版上，5,250 条错误样本里有
79 条的估计落在 `δ` 之内——满额采纳时误验率 1.50%，只有旧目标的三分之一。于是 `fit_tau`
返回了最低的候选分数，没有任何样本弃权，coverage 恒为 1.000，`ρ` 没有参与过任何一次判决。
**一个任何方法都无法触碰的目标标定不了任何东西**，它只会让三态判决实际退化成两态，同时看
起来做了更多事。

这个区分不是学究式的。早先的实现把每个**被采纳**的错误样本都算作误验，等于让一个方法为自己
正确的拒绝付出代价。在十张切片的冻结版的 LoFTR 运行上实测：在 `ρ` 的上限处，被采纳率是 10.3%
而被接受率是 0.00%，于是 5% 的目标在任何阈值下都不可达，`fit_tau` 返回了一个高于所有分数的
值，一个几乎完美拒绝 fault 的验证器最终在 100% 的样本上弃权。

在出厂的三个 freeze 里，`calibration` 只包含两名患者。用这么少的患者拟合出的
`τ`/`δ` 说明的是整条流水线——拟合、冻结、打分——是可以跑通的；并不能说明这些阈值
在更大人群上具有代表性意义上的「校准」。另外要注意，如果给 `fit_tau` 的 calibration
split 里不含任何有 fault 的 item，它会直接抛出异常：衡量误验证率至少需要一个错误的
claim 作为分母，一个在全干净 split 上拟合出的 `τ` 会把一切都判为可验证，验证器就永远
不会弃权。请把从 `test` 上得到的数字如实报告为——在出厂 freeze 上对这套协议的一次
演示——而不是关于某个验证器真实世界可靠性的结论。

**读 `balanced_accuracy` 之前先看 `δ`。** `thresholds.json` 记录了 clean item 残差分布的
p50/p75/p90/p95/p99(`residual_report`,`pyrafault/verify/thresholds.py`),`δ` 就是其中的
p99。当一个方法的定位有很重的失败尾巴时,这个 p99 由尾巴决定而不是由典型残差决定;这么宽的
`δ` 会让 `decide`(§7)里的空间检验近乎形同虚设:`d <= δ · query_side` 对几乎任何估计都成立,
`decide` 退化成「`ρ` 是否越过 `τ`」,`balanced_accuracy` 也就不再衡量 `ℓ̂` 是否与 `ℓ` 一致。

两个随仓库提供的 matcher 恰好落在两边。在十张切片的冻结版 calibration 的 clean item 上实测
(各 300 条):

| matcher | p50 | p75 | p90 | p95 | p99 = `δ` | 残差 > 1 个 query 边长 |
|---|---|---|---|---|---|---|
| LoFTR | 0.004 | 0.006 | 0.011 | 0.014 | 0.039 | 0.0% |
| SIFT  | 0.004 | 0.911 | 4.06 | 7.97 | 10.73 | 23.7% |

LoFTR 的 `δ` 不到 query 边长的二十分之一,空间检验有区分力;SIFT 的超过十个边长,没有。所以
在相信 `balanced_accuracy` 之前,打开 `thresholds.json` 比较 p99 和 p50。`verification_score`
的 AUROC 与 AUPRC(§8)基于连续分数而不是 `decision` 计算,完全不依赖 `δ`。
