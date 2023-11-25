'''
    對原本的資料做preprocessing, 多增加一些特徵幫助模型判斷
'''
import pandas as pd

def preprocessing(data, top15_risk_stocn, top5_risk_hour, category):
    ''' 
        input:
            data: 欲處理的資料
        新增特徵
        loctm_hour: 將時間以小時做間隔
        mcc_10: 除了交易量前十大mcc保持一樣外其他設others
        stocn_10: 除了交易量前十大stocn保持一樣外其他設others
        hour_accu: 交易時間以累計小時數表示
        time_diff: 同一個顧客前一筆交易與此筆交易的間隔小時數
        one_hour: 同一個顧客前一筆交易與此筆交易的間隔時間小於一小時
        high_risk_stocn: 盜刷數量前15大消費地國別
        high_risk_hour: 盜刷數量前5大交易時間點
    '''
    data['loctm_hour'] = data['loctm'] // 10000
    top_ten_mcc = data['mcc'].value_counts().head(10).index.tolist()
    data['mcc_10'] = data['mcc'].apply(lambda x: x if pd.isnull(x) or x in top_ten_mcc else 999)
    top_ten_stocn = data['stocn'].value_counts().head(10).index.tolist()
    data['stocn_10'] = data['stocn'].apply(lambda x: x if pd.isnull(x) or x in top_ten_stocn else 999)
    data['hour_accu'] = data['locdt']*24 + data['loctm_hour'] 
    data['time_diff'] = data.groupby('chid')['hour_accu'].diff()
    data['one_hour'] = data['time_diff'].apply(lambda x: 1 if pd.notnull(x) and x == 0 else 0)
    data['high_risk_stocn'] = data['stocn'].apply(lambda x: 1 if x in top15_risk_stocn else 0)
    data['high_risk_hour'] = data['loctm_hour'].apply(lambda x: 1 if x in top5_risk_hour else 0)
    data[category] = data[category].astype(str)
