# 美团外卖配送超时预测与调度效率分析

本项目基于美团外卖即时配送调度数据，分析订单超时风险，并构建机器学习模型预测订单是否会超时。

项目目标是通过订单、骑手、时间和距离等派单时刻可获得的信息，识别高风险订单，为配送调度优化和超时预警提供参考。

## 项目背景

在外卖即时配送场景中，订单是否能够按预计时间送达，直接影响用户体验和平台履约效率。如果平台能够提前识别可能超时的订单，就可以在派单、骑手调度和风险干预上做出更及时的决策。

本项目要解决的问题是：

> 能否基于派单时刻的订单信息、骑手状态、时间特征和距离特征，预测一笔订单最终是否会超时？

目标变量定义如下：

```text
late = 1，表示 arrive_time > estimate_arrived_time，即订单超时
late = 0，表示订单未超时
```

因此，本项目是一个二分类预测任务。

## 数据说明

项目使用美团外卖调度相关 CSV 数据文件。运行代码前，需要将以下四个原始数据文件放入 `data/` 目录：

```text
data/
  all_waybill_info_meituan_0322.csv
  courier_wave_info_meituan.csv
  dispatch_rider_meituan.csv
  dispatch_waybill_meituan.csv
```

由于原始数据文件通常较大，且可能包含课程数据集信息，本仓库不上传 `data/` 原始数据。

建模主要使用订单运单表和骑手状态表。订单表中包含订单时间、派单时间、骑手接单时间、取餐时间、送达时间、预计送达时间、商家和顾客位置等信息；骑手状态表用于补充骑手在派单时刻的负载情况。

## 分析流程

项目整体流程包括：

1. 数据读取与清洗
2. 目标变量构造
3. 特征工程
4. 描述性分析
5. 模型训练与评估
6. 结果输出与可视化

## 描述性分析

清洗后用于建模的数据量约为：

```text
有效订单数：532,620
订单超时率：14.46%
平均总配送时长：31.00 分钟
平均派单等待时间：4.09 分钟
平均商家到顾客距离：1.55 公里
```

从派单等待时间看，等待时间越长，订单超时风险整体越高：

```text
0-1 分钟：超时率约 9.55%
1-2 分钟：超时率约 13.86%
2-5 分钟：超时率约 15.35%
5-10 分钟：超时率约 20.48%
10-20 分钟：超时率约 23.70%
```

从商家到顾客距离看，距离越远，超时率越高：

```text
最近距离分组：超时率约 7.06%
最远距离分组：超时率约 19.93%
```

这说明配送距离、派单等待时间和配送时段都与超时风险存在明显关系。

## 特征工程

模型使用的主要特征包括：

```text
dispatch_wait_min：派单等待时间
promised_remaining_min：距离承诺送达还剩多少时间
prep_remaining_min：距离预计备餐完成还剩多少时间
estimated_prep_duration_min：预计备餐时长
merchant_to_customer_km：商家到顾客距离
courier_to_merchant_km：骑手到商家距离
courier_to_customer_km：骑手到顾客距离
dispatch_hour：派单小时
dispatch_dayofweek：星期几
is_weekend：是否周末
is_prebook：是否预订单
da_id：配送区域
onhand_order_count：骑手当前手上订单数
```

其中，距离类特征通过经纬度和 haversine 公式计算得到；时间类特征通过订单各阶段时间戳差值计算得到。

## Baseline

本项目设置了两个层面的 baseline：

第一类是简单多数类 baseline。由于超时订单占比约为 14.46%，如果所有订单都预测为“不超时”，整体准确率大约为 85.54%。但这种方法完全无法识别超时订单，超时类召回率为 0，因此业务价值较低。

第二类是机器学习 baseline。项目使用 Logistic Regression 作为基础模型，它结构简单、解释性强，可以作为和更复杂模型对比的基准。

## 模型方法

项目训练并比较了两个模型：

- Logistic Regression：解释性较强，作为基础分类模型
- Random Forest：能够捕捉非线性关系和特征交互，适合处理配送场景中的复杂影响因素

由于超时订单占比较低，模型训练时使用了类别权重，以缓解类别不平衡问题。

## 评价指标

模型使用以下指标进行评估：

- Accuracy：整体预测正确率
- Precision：预测为超时的订单中，真正超时的比例
- Recall：真实超时订单中，被模型识别出来的比例
- F1-score：Precision 和 Recall 的综合指标
- ROC-AUC：模型区分超时与非超时订单的能力

由于本项目关注超时风险预警，因此相比单纯的 Accuracy，更关注 Recall 和 ROC-AUC。

## 实验结果

模型实验结果如下：

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.5758 | 0.2077 | 0.7055 | 0.3210 | 0.6773 |
| Logistic Regression | 0.6121 | 0.2117 | 0.6351 | 0.3175 | 0.6731 |

Random Forest 的 ROC-AUC 略高，并且对超时订单的召回率更高，能够识别约 70.55% 的真实超时订单。因此在超时风险预警场景下，Random Forest 更适合作为主要模型。

特征重要性排名靠前的变量包括：

1. 商家到顾客距离
2. 派单等待时间
3. 骑手到顾客距离
4. 距离承诺送达剩余时间
5. 骑手到商家距离

这些结果说明，订单超时主要受到配送距离、派单等待、骑手位置和剩余履约时间影响。

## 项目结构

```text
.
|-- meituan_late_prediction.py
|-- requirements.txt
|-- README.md
|-- .gitignore
`-- outputs/
    |-- model_metrics.csv
    |-- summary_metrics.csv
    |-- feature_importance_top15.csv
    |-- late_rate_by_hour.csv
    |-- late_rate_by_dispatch_wait_bin.csv
    |-- late_rate_by_distance_bin.csv
    `-- classification reports
```

运行脚本后，还会在本地生成 `figures/` 目录，用于保存描述性分析图、混淆矩阵、ROC 曲线和特征重要性图。

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

将原始 CSV 文件放入 `data/` 目录后运行：

```bash
python meituan_late_prediction.py --data_dir data --output_dir outputs --fig_dir figures
```

运行结束后，结果表格会保存在 `outputs/` 目录，图片会保存在 `figures/` 目录。

## 项目结论

本项目基于美团外卖调度数据，构建了订单超时风险预测模型。描述性分析表明，派单等待时间、配送距离和派单时段会明显影响订单超时率。模型实验中，Random Forest 相比 Logistic Regression 取得了略高的 ROC-AUC 和更高的超时订单召回率，更适合作为超时风险预警模型。

项目结果可以为平台在派单优化、骑手调度和高风险订单提前干预方面提供参考。
