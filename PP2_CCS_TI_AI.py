import pandas as pd
df=pd.read_csv('PP2_DATA_CSS_AI_TI.csv')
df.head()
import matplotlib.pyplot as plt
X=df.iloc[:,:24]
y=df.iloc[:,24:27]
X.head()
y.head()
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)
feat_importances=pd.Series(model.feature_importances_,index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=26)
from sklearn.ensemble import RandomForestRegressor
rf_random=RandomForestRegressor()
import numpy as np
n_estimators=[int(X) for X in np.linspace(start=100, stop=1200, num=12)]
print(n_estimators)
max_features=['auto','sqrt']
max_depth=[int(X) for X in np.linspace(5,30, num=6)]
min_samples_split=[2,5,10,15,100]
min_samples_leaf=[1,2,5,10]
from sklearn.model_selection import RandomizedSearchCV
random_grid={'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth':max_depth,
            'min_samples_split':min_samples_split,
            'min_samples_leaf':min_samples_leaf}
print(random_grid)
rf=RandomForestRegressor()
rf_random=RandomizedSearchCV(estimator=rf, param_distributions=random_grid, scoring='neg_mean_squared_error',n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
rf_random.best_score_
import seaborn as sns
predictions=rf_random.predict(X_test)
sns.distplot(y_test-predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
r2=metrics.r2_score(y_test, predictions)
print(r2)
import pickle
file=open('PP2_CCS_TI_AI3.pkl','wb')
pickle.dump(rf_random, file)
#with open('PP2_CCS_TI_AI.pkl','rb') as f:
    #data=pickle.load(f)
#print(data.predict(np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])))
