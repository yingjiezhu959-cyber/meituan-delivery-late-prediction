# Preliminary Result Summary

## Sample and Descriptive Metrics

```csv
,value
n_model_records,532620.0
late_rate,0.1446321955615636
mean_total_delivery_min,31.001921069430363
mean_dispatch_wait_min,4.089433617463357
mean_merchant_to_customer_km,1.5478792464759086
rider_state_match_rate,3.7550223423829372e-06
```

## Model Metrics

```csv
model,accuracy,precision,recall,f1,auc
random_forest,0.57584,0.20772546419098142,0.7055180180180181,0.320952868852459,0.6773162452829554
logistic_regression,0.61208,0.21167198348658284,0.6351351351351351,0.3175228712174525,0.6730512683515294
```

The current models use features available at dispatch time to predict whether an order will be late. Random Forest achieves a slightly higher ROC-AUC and a higher recall for late orders, while Logistic Regression provides a simple and interpretable baseline.
