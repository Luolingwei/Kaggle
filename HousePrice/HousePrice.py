from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.width',None)

#准备数据
raw1=pd.read_csv('train.csv')
# raw2=pd.read_csv('test.csv')
raw1.fillna(0,inplace=True)
# raw2.fillna(0,inplace=True)
features=['LotArea','YearBuilt','OverallQual','OverallCond','GrLivArea','GarageArea']
train_x,test_x,train_y,test_y=train_test_split(raw1[features],raw1['SalePrice'],test_size=0.2)
# test_x=raw2[features]
train_x=np.log1p(train_x)
train_y=np.log1p(train_y)
test_x=np.log1p(test_x)

#寻找最佳参数
clf1=GradientBoostingRegressor()
param_dist={'n_estimators':[300,500,800,1000],'min_samples_leaf':[20,50,100],'min_samples_split':[100,300,800]}
grid_search=GridSearchCV(clf1,param_dist,scoring='neg_mean_squared_error',cv=5)
grid_search.fit(train_x,train_y)
print(grid_search.best_params_)

clf2=XGBRegressor()
param_dist={'n_estimators':[300,500,800,1000,2000],'learning_rate':[0,1,0.2,0.5,0.8],'gamma':[0,1,10,50,100],
            'min_child_weight':range(1,6,2),'subsample':[i/100 for i in range(75,90,5)],'max_depth':[1,10,20,100,500]}
grid_search=GridSearchCV(clf2,param_dist,scoring='neg_mean_squared_error',cv=5)
grid_search.fit(train_x,train_y)
print(grid_search.best_params_)

clf3=RandomForestRegressor()
param_dist={'n_estimators':[300,500,800,1000],'max_features':[1,2,3,4,5,6],'min_samples_leaf':[20,50,100],
            'min_samples_split':[100,200,300],'max_depth':[1,10,20,100]}
grid_search=GridSearchCV(clf3,param_dist,scoring='neg_mean_squared_error',cv=5)
grid_search.fit(train_x,train_y)
print(grid_search.best_params_)

#模型训练
GBDT=GradientBoostingRegressor(n_estimators=300,min_samples_leaf=20,min_samples_split=100)
GBDT.fit(train_x,train_y)
Prediction1=GBDT.predict(test_x)
Prediction1=np.expm1(Prediction1)
print(r2_score(test_y,Prediction1))

XGB=XGBRegressor()
XGB.fit(train_x,train_y)
Prediction2=XGB.predict(test_x)
Prediction2=np.expm1(Prediction2)
print(r2_score(test_y,Prediction2))

RF=RandomForestRegressor()
RF.fit(train_x,train_y)
Prediction3=RF.predict(test_x)
Prediction3=np.expm1(Prediction3)
print(r2_score(test_y,Prediction3))

Prediction=0.2*Prediction1+0.6*Prediction2+0.2*Prediction3
print(r2_score(test_y,Prediction))

#精度可视化
error=Prediction-test_y
print(error.sum())
plt.figure(figsize=(15,7))
plt.subplot(2,1,2)
plt.plot(np.arange(len(error)),error,label='error')
plt.legend()
plt.subplot(2,1,1)
plt.plot(np.arange(len(Prediction)),Prediction,'ro--',label='Prediction')
plt.plot(np.arange(len(test_y)),test_y,'ko--',label='Real')
plt.legend()
plt.show()

# 数据输出
# submission=pd.DataFrame({'Id':test.Id,'SalePrice':Prediction})
# submission.to_csv('C:/Users/asus/Desktop/submission.csv',index=False)
