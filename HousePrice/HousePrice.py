from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
features=['LotArea','YearBuilt','OverallQual','1stFlrSF','FullBath']
train_x=train[features]
train_y=train['SalePrice']
predict_x=test[features]

RF=RandomForestRegressor(n_estimators=2,n_jobs=-1)
RF.fit(train_x,train_y)

results=RF.predict(predict_x)
print(results)

submission=pd.DataFrame({'Id':test.Id,'SalePrice':results})
submission.to_csv('submission.csv',index=False)
