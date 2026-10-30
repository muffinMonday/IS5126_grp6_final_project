## 1 文件总览

| 文件 | 是否必需 | 内容与用途 | 主要关联方式 |
|---|---|---|---|
| `comments_clean.csv` | 使用数据时必需 | 冻结的10,000条评论，含原文、话题清理文本、UTC时间和抽样信息；作为建模及结果解释的输入 | `comment_id`；按 `month` 汇总 |
| `sampling_manifest.json` | 建议随数据保留；版本校验时必需 | 来源、采样方法、种子、截止时间、数量、局限性及文件SHA-256 | 校验交付CSV是否属于同一语料版本 |
| `monthly_coverage.csv` | 建议随数据保留；月度分析时必需 | 各月抽样数量、候选数量、缺口、月份边界和不完整月份标记 | 标准化后的 `month` |
| `comment_topics.csv` | 比较现有K-means或整合分类结果时必需 | 每条输入评论的簇编号、建议名称、分配状态及模糊度 | 与评论、LDA结果或毒性预测按 `comment_id` 一对一合并 |
| `topic_profiles.csv` | 解释主题或比较方法时建议保留 | 每簇的关键词、规模、建议名称、语义解释和限制 | 按 `cluster` 多对一关联，不能按此表行号关联 |

建议额外保留同一结果目录中的 `run_parameters.json`。它记录PCA维数、k、初始化次数、输入哈希和结果文件哈希，可以证明评论分配与主题画像来自哪一次模型运行。仅凭5个文件的名称相同，不能证明它们属于同一个版本。

## 2 当前数据版本与范围

| 项目 | 当前版本 |
|---|---|
| 社区与来源 | r/jobs；Arctic Shift公共历史档案 |
| 样本期间 | 2026年1—10月；10月不完整 |
| 冻结时间 | 2026-10-28 07:10:54 UTC，即新加坡时间15:10:54 |
| 抽样 | 每月20个等长UTC时间段，每段随机起点读取有限候选，清洗后无放回抽50条 |
| 月度配额 | 每月1,000条，共10,000个唯一 `comment_id`；种子42 |
| 聚类输入 | 排除106条明确版务模板和124条零TF-IDF向量后，共9,770条 |
| 当前K-means方案 | TF-IDF → 中心化PCA100 → 逐行L2归一化 → K-means，k=18、n_init=30、种子42 |
| 月度聚类分母 | 当月 `topic_status == "assigned"` 的评论数，实际为960—985条，而非固定1,000条 |

## 3 各文件字段说明

### 3.1 comments_clean.csv

一行是一条抽样评论，当前共有10,000行。

| 字段 | 含义与使用建议 |
|---|---|
| `comment_id` | 评论唯一ID，作为字符串读取；跨方法、跨预测文件关联的主键 |
| `text` | 原始正文；用于阅读、解释及传给使用自身训练预处理的毒性分类器，不要覆盖 |
| `text_topic` | 话题分析清理文本；已做HTML实体解码、小写及共享清理，可作为后续话题向量化输入 |
| `created_utc` | 评论发表时间，UTC Unix秒，不是采集时间或毫秒 |
| `created_at` | 评论发表时间的UTC日期表示 |
| `month` | 按UTC划分的月份，格式为 `YYYY-MM`，例如 `2026-01` |
| `subreddit` | 来源社区，当前为 `jobs` |
| `link_id`、`parent_id` | 所属帖子和父节点；`link_id` 可用于检查同一热门帖子是否贡献过多评论 |
| `permalink` | 依据来源ID构造的评论链接，未逐一访问验证 |
| `collected_utc` | 采集或映射记录的时间，不应用它划分评论月份 |
| `sampling_stratum` | 月内抽样时间段编号，本次为0—19，每月每段50条 |
| `word_count` | 清理后的字母词数量，本次至少3个 |
| `is_bot`、`language`、`language_status` | 来源机器人与语言信息；语言可能未验证，不能据此假定全部正文都是英语 |
| `duplicate_text`、`normalized_text_key` | 相同规范文本的审计信息；不同ID的相同正文被保留，不要擅自全局去重 |
| `source_row`、`exclusion_reason` | 来源位置和清理审计字段；本文件中的评论已经通过数据准备清洗，`exclusion_reason` 通常为空 |

`text_topic` 不是TF-IDF向量，也没有包含PCA结果。停用词处理等建模规则仍在向量化阶段执行。不要把话题模型的额外停用词处理强加给已训练的毒性分类器。

### 3.2 sampling_manifest.json

| 字段或字段组 | 含义 |
|---|---|
| `year`、`as_of_utc` | 采样年份及冻结截止时间 |
| `seed`、`target_per_month`、`min_words`、`strata_per_month` | 随机种子、月配额、清理最短词数及月内分层数量 |
| `source` | 数据来源、接口、请求次数及采集记录位置 |
| `sampling`、`population`、`limitations` | 抽样设计、所覆盖总体和明确限制；报告中应保留这些限制 |
| `counts` | 原始候选20,213条、合格17,766条、排除2,447条、最终抽样10,000条 |
| `sha256` | 文件字节校验值，用于判断文件是否在准备后发生变化 |

manifest还引用未包含在这5个文件内的候选池、排除记录等。这不影响读取已交付的5文件，但不能把“未交付审计文件”理解成不存在。下面的快速示例仅校验实际使用的文件。

### 3.3 monthly_coverage.csv

一行是一个月份，当前共10行。

| 字段 | 含义 |
|---|---|
| `month` | 月份；若被表格软件改为 `Jan-26` 等格式，须先校验并规范为 `YYYY-MM` 再关联 |
| `window_start_utc`、`window_end_utc` | 该月实际采样时间窗，末月截止于冻结时点 |
| `partial_month` | 是否为不完整月份；10月为True，不应与完整月份直接等同分析 |
| `target` | 月度目标数，当前为1,000 |
| `eligible_pool` | 该月读取候选中通过清理的数量，不是全月所有评论数 |
| `selected` | 最终抽样数量，当前每月1,000；不是进入K-means的数量 |
| `shortfall` | 目标与实际抽样的缺口，当前各月为0 |
| `source_window_complete` | 是否完整读取来源窗口；当前False反映有限候选设计，不代表该月没有数据或请求失败 |
| `status` | 抽样状态；`target_met` 表示达到配额，不表示全档案已读取完毕 |

**False字符串不能直接用 `bool("False")` 转换**，因为Python会将非空字符串视为True。应明确映射true/false。也不要把覆盖表中的 `Jan-26` 与评论表中的 `2026-01` 直接合并。

当前覆盖表曾被表格软件改变月份显示和布尔值大小写。仓库的 `verified_coverage` 仅在能重建原manifest哈希时，接受这种格式变化，并在内存恢复；数值变化仍会报错。不要通过Excel另存CSV来“修复”原始文件。

### 3.4 comment_topics.csv

一行对应一条输入评论，仍为10,000行，而非只保留9,770条已分配评论。

| 字段 | 含义与注意事项 |
|---|---|
| `comment_id`、`month` | 关联主键和UTC月份，应与评论文件一致 |
| `cluster` | 当前K-means簇编号0—17；未分配评论为空。它是类别ID，不是等级、概率或毒性标签 |
| `topic_label` | 英文建议主题名；不是人工标注的真实类别 |
| `topic_status` | `assigned`、`moderation_notice` 或 `zero_tfidf`，具体数量见下表 |
| `assignment_margin` | 最近和次近质心距离的相对差，近0表示分配较模糊；不是正确概率 |
| `semantic_confidence` | 语义可解释性定性评级，如 `Medium`、`High`，不是模型置信度 |
| `label_source`、`human_validated` | 名称来源及是否经人工验证；当前建议名为 `assistant_suggested`，人工验证标志为False |
| `focus_topic` | 是否为报告选取的六个案例，即C2、C5、C8、C9、C11、C17；不是“其他簇无效”的标记 |

| `topic_status` | 数量 | 使用方式 |
|---|---:|---|
| `assigned` | 9,770 | 可用于现有簇占比计算、共同样本比较和主题级预测整合 |
| `moderation_notice` | 106 | 明确版务通知模板，未进入话题聚类；保留ID供审计 |
| `zero_tfidf` | 124 | 在现有词表下没有有效特征，未进入聚类；保留ID供审计 |

读取后 `cluster` 可能显示为 `2.0` 等浮点形式；这只是CSV空值导致的类型表示，可转为Pandas可空整数 `Int64`。空值不能填成0，否则会误并入C0。也不要按CSV行号或导出顺序拼接其他模型结果。

### 3.5 topic_profiles.csv

一行是一个簇，当前共18行。

| 字段或字段组 | 含义 |
|---|---|
| `cluster`、`topic_label` | 簇编号和建议名，应与同目录评论分配表一致 |
| `n_comments`、`share` | 全样本中的簇规模及占9,770条已分配评论的比例；包含不完整10月，不是单月占比 |
| `top_terms` | 簇内原始TF-IDF均值最高的词或双词组，用于理解主题 |
| `distinctive_terms` | 簇内均值减全局均值后的突出词，作为解释辅助 |
| `semantic_confidence`、`semantic_evidence` | 定性可解释性评级及依据；不代表分类准确率 |
| `employment_topic`、`interpretation_limit` | 是否可视为就业议题及其解释限制；回复形式明确不等于就业主题明确 |
| `trend_judgment`、`robustness_assessment` | 当前冻结样本的趋势解释及敏感性判断，不是可用于新模型的真实标签 |
| `project_relevance` | 该案例与项目研究问题的联系 |
| `label_reviewed`、`label_source`、`human_validated` | 审阅与来源标志；当前建议名称仍待人工验证 |
| `focus_topic`、`chart_label` | 案例展示标志及图表标签，不影响原始模型分配 |

C7和C12等混合簇仍是已分配结果的一部分。Notebook热力图仅隐藏C7的显示，**没有删除其评论，也没有从占比分母移除C7**。复核当前结果时不要自行过滤这些簇。
