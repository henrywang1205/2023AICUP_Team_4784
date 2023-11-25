# 模型
本資料夾含有一個程式:

* train.py: 訓練模型
* Catboost 參數設定
* learning_rate=0.3
* depth=12
* cat_features=category 設定類別屬性column
* task_type="GPU"  GPU運算大幅提升速度
* iterations=100
* verbose=10 
* eval_metric='F1'
* random_seed=seed
* early_stopping_rounds=50
* use_best_model=True