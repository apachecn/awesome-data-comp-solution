# 开源来自：知识星球-Kaggle数据竞赛免费版，转载请注明出处

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

train = pd.read_csv('first_round_training_data.csv')
test = pd.read_csv('first_round_testing_data.csv')

features = ["Parameter1","Parameter2","Parameter3","Parameter4","Parameter5","Parameter6","Parameter7","Parameter8","Parameter9","Parameter10"]

def encoder(x):
    return {'Excellent':0,'Good':1,'Pass':2,'Fail':3}[x]

train['label'] = train.Quality_label.apply(encoder)
train['label_Excellent'] = 1*(train['label'] == 0)
train['label_Good'] = 1*(train['label'] == 1)
train['label_Pass'] = 1*(train['label'] == 2)
train['label_Fail'] = 1*(train['label'] == 3)

model = GradientBoostingClassifier(max_depth=3,learning_rate=0.1,n_estimators=100,random_state=2019)
model.fit(train.loc[:,features],train.label)

test['prediction'] = model.predict(test.loc[:,features])
test['prob_Excellent'] = 0.0
test['prob_Good'] = 0.0
test['prob_Pass'] = 0.0
test['prob_Fail'] = 0.0
test.loc[:,['prob_Excellent','prob_Good','prob_Pass','prob_Fail']] = model.predict_proba(test.loc[:,features])


prediction = test.groupby(['Group'],as_index=False)['prob_Excellent','prob_Good','prob_Pass','prob_Fail'].mean()
prediction.columns = ['Group','Excellent ratio','Good ratio','Pass ratio','Fail ratio']
prediction.to_csv('baseline.csv',index=False)