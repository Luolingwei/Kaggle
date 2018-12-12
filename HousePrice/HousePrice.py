from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#数据准备
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
train.fillna(0,inplace=True)
test.fillna(0,inplace=True)
features=['LotArea','YearBuilt','OverallCond','1stFlrSF','TotalBsmtSF','2ndFlrSF','GrLivArea','GarageArea']
train_x=train[features]
train_y=train['SalePrice']
train_x=np.log1p(train_x)
train_y=np.log1p(train_y)
test_x=test[features]
test_x=np.log1p(test_x)

#寻找最优参数

#模型训练
RF=RandomForestRegressor(n_estimators=3000,n_jobs=-1)
RF.fit(train_x,train_y)
Prediction1=RF.predict(test_x)
Prediction1=np.expm1(Prediction1)

GBDT=GradientBoostingRegressor(n_estimators=2000,max_depth=10)
GBDT.fit(train_x,train_y)
Prediction2=GBDT.predict(test_x)
Prediction2=np.expm1(Prediction2)

XGB=xgb.XGBRegressor()
XGB.fit(train_x,train_y)
Prediction3=XGB.predict(test_x)
Prediction3=np.expm1(Prediction3)

Prediction=0.2*Prediction1+0.3*Prediction2+0.5*Prediction3

#数据输出
submission=pd.DataFrame({'Id':test.Id,'SalePrice':Prediction})
submission.to_csv('submission.csv',index=False)
