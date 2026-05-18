# Meituan Delivery Lateness Prediction

This project analyzes Meituan food delivery dispatch data and builds machine learning models to predict whether an order will be delivered late.

The goal is to support delivery risk warning and dispatch efficiency analysis by identifying orders with high lateness risk at dispatch time.

## Project Objective

In instant food delivery, late orders directly affect customer experience and platform efficiency. This project focuses on the following question:

> Can we predict whether an order will be late using dispatch-time order, rider, time, and distance features?

The target variable is:

```text
late = 1, if arrive_time > estimate_arrived_time
late = 0, otherwise
```

This is a binary classification task.

## Dataset

The project expects four Meituan dispatch-related CSV files under the `data/` directory:

```text
data/
  all_waybill_info_meituan_0322.csv
  courier_wave_info_meituan.csv
  dispatch_rider_meituan.csv
  dispatch_waybill_meituan.csv
```

The raw data files are not included in this repository. Please place them in the `data/` directory before running the code.

The main modeling data comes from order waybill information and rider dispatch state information. The features include order timestamps, dispatch timestamps, rider/order status, merchant and customer coordinates, and rider workload information.

## Methodology

The pipeline includes:

1. Data loading and cleaning
2. Target variable construction
3. Feature engineering
4. Descriptive analysis
5. Model training and evaluation
6. Result visualization

## Feature Engineering

The main features used in the prediction model include:

```text
dispatch_wait_min
promised_remaining_min
prep_remaining_min
estimated_prep_duration_min
merchant_to_customer_km
courier_to_merchant_km
courier_to_customer_km
dispatch_hour
dispatch_dayofweek
is_weekend
is_prebook
da_id
onhand_order_count
```

Distance features are calculated from geographic coordinates using the haversine formula. Time features are calculated from order, dispatch, meal preparation, and delivery timestamps.

## Models

Two models are trained and compared:

- Logistic Regression: used as an interpretable baseline model
- Random Forest: used to capture nonlinear relationships and feature interactions

Class weights are used to reduce the impact of class imbalance, since late orders account for a smaller proportion of all orders.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Since this is a late-order risk prediction task, recall and ROC-AUC are especially important. A useful model should identify as many truly late orders as possible.

## Current Results

After cleaning and feature construction, the modeling dataset contains approximately 532,620 valid orders.

Summary statistics:

```text
Late order rate: 14.46%
Average total delivery time: 31.00 minutes
Average dispatch waiting time: 4.09 minutes
Average merchant-customer distance: 1.55 km
```

Model performance:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.5758 | 0.2077 | 0.7055 | 0.3210 | 0.6773 |
| Logistic Regression | 0.6121 | 0.2117 | 0.6351 | 0.3175 | 0.6731 |

The Random Forest model achieves the best ROC-AUC and a higher recall for late orders.

Top important features include:

1. Merchant-to-customer distance
2. Dispatch waiting time
3. Courier-to-customer distance
4. Remaining promised delivery time
5. Courier-to-merchant distance

## Repository Structure

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

The Python script can also generate visualization files under `figures/` after the raw data is placed locally and the pipeline is run.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the raw CSV files under `data/`, then run:

```bash
python meituan_late_prediction.py --data_dir data --output_dir outputs --fig_dir figures
```

The script will generate summary tables under `outputs/` and visualizations under `figures/`.

## Key Takeaways

The descriptive analysis and model results show that late delivery risk is strongly related to dispatch waiting time, delivery distance, courier location, and remaining promised delivery time. These results can help support dispatch optimization and early warning for high-risk orders.
