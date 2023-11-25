'''
    使用catboost模型做training
'''
import pandas as pd
from sklearn.utils import shuffle
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from Preprocess import preprocess

#讀取train_path的資料集作為模型訓練的輸入，讀取predict_path的資料進行預測，輸出預測結果至result_path
def train(train_path, predict_path, result_path):
    '''
        input:
            train_path: training csv path
            predict_path: the csv file data for prediction
            result_path: the output result csv path
    '''
    train = pd.read_csv(train_path)
    seed = 15

    top15_risk_stocn = train[train['label'] == 1]['stocn'].value_counts().head(15).index.tolist()
    train['loctm_hour'] = train['loctm'] // 10000
    top5_risk_hour = train[train['label'] == 1]['loctm_hour'].value_counts().head(5).index.tolist()
    category = ['loctm_hour','one_hour','contp','etymd','mchno','acqic','mcc','mcc_10','ecfg','insfg','bnsfg','stocn','stocn_10','scity','stscd','ovrlt','flbmk','hcefg','csmcu','flg_3dsmk',
                'high_risk_stocn','high_risk_hour','chid','cano']
    preprocess.preprocessing(train, top15_risk_stocn, top5_risk_hour, category)

    train = shuffle(train, random_state=seed)
    selection=['loctm_hour','time_diff','one_hour','locdt','loctm','mchno','acqic','contp','etymd','mcc','mcc_10','conam','ecfg','insfg','iterm','bnsfg','flam1','stocn','stocn_10','scity','stscd',
            'ovrlt','flbmk','hcefg','csmcu','csmam','flg_3dsmk','high_risk_stocn','high_risk_hour','chid','cano']
    X = train[selection]
    y = train[["label"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.1)
    model = CatBoostClassifier(learning_rate=0.3, depth=12, cat_features=category, task_type="GPU", iterations=100, verbose=10, eval_metric='F1',random_seed=seed, early_stopping_rounds=50, use_best_model=True)
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    output = pd.read_csv(predict_path)
    preprocess.preprocessing(output, top15_risk_stocn, top5_risk_hour, category)
    xt = output[selection]
    ot = model.predict(xt, task_type="CPU")
    out = output[["txkey"]]
    out["pred"] = 0
    for i in range(0, len(out)):
        if(ot[i]==1):
            out.loc[i,"pred"] = 1
    out.to_csv(result_path, index=False)