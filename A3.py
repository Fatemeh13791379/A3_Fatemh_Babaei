# A3_Fatemh_Babaei
#This is the third project for the AI course at Gham Lab.
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = fetch_california_housing()
features = data.data  
target = data.target 
data_New = pd.DataFrame(features, columns=data.feature_names)
data_New['MedHouseValue'] = target
data_New.info()
print(data.feature_names)
###['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

print(data.target_names) 
##['MedHouseVal']
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20640 entries, 0 to 20639
Data columns (total 9 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   MedInc         20640 non-null  float64
 1   HouseAge       20640 non-null  float64
 2   AveRooms       20640 non-null  float64
 3   AveBedrms      20640 non-null  float64
 4   Population     20640 non-null  float64
 5   AveOccup       20640 non-null  float64
 6   Latitude       20640 non-null  float64
 7   Longitude      20640 non-null  float64
 8   MedHouseValue  20640 non-null  float64
dtypes: float64(9)
memory usage: 1.4 MB
'''
data_New.describe()
'''

  MedInc      HouseAge  ...     Longitude  MedHouseValue
count  20640.000000  20640.000000  ...  20640.000000   20640.000000
mean       3.870671     28.639486  ...   -119.569704       2.068558
std        1.899822     12.585558  ...      2.003532       1.153956
min        0.499900      1.000000  ...   -124.350000       0.149990
25%        2.563400     18.000000  ...   -121.800000       1.196000
50%        3.534800     29.000000  ...   -118.490000       1.797000
75%        4.743250     37.000000  ...   -118.010000       2.647250
max       15.000100     52.000000  ...   -114.310000       5.000010

[8 rows x 9 columns]
'''
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LinearRegression()
my_params = {'fit_intercept':[True],'copy_X': [True, False]}
gs = GridSearchCV(model, my_params, cv=kf, scoring='neg_mean_absolute_percentage_error')
gs.fit(features, target)
best_score = gs.best_score_
print("LR:", best_score)####LR: -0.3175876329794739

##پیش‌بینی‌های مدل به طور متوسط 31% از مقادیر واقعی فاصله دارند.یعنی 69 % دقت دارد .
best_params = gs.best_params_ 
print("LR:", best_params)
##LR: {'copy_X': True, 'fit_intercept': True}

##KNN

model = KNeighborsRegressor()
my_params = { 'n_neighbors': [1, 2, 3, 4, 5, 6, 10],'metric': ['minkowski', 'euclidean', 'manhattan']}
gs = GridSearchCV(model, my_params, cv=kf, scoring='neg_mean_absolute_percentage_error')
gs.fit(features, target)

best_score = gs.best_score_
print(" KNN:", best_score) ### KNN: -0.4847991767838546 .پیش‌بینی‌های مدل به طور متوسط 48% از مقادیر واقعی فاصله دارند.یعنی 52 % دقت دارد .
best_params = gs.best_params_ 
print("KNN:", best_params)###KNN: {'metric': 'manhattan', 'n_neighbors': 4}

###مقدار MAPE برای مدل رگرسیون خطی بهتر از مدل KNN است. یعنی مدل رگرسیون خطی پیش‌بینی‌های دقیق‌تری ارائه می‌دهد.



###DT
DT_model = DecisionTreeRegressor(random_state=42)
DT_my_params = { 'max_depth': [1, 2, 3, 4, 5, 6, 7, 10]} 

gs = GridSearchCV(model, my_params, cv=kf, scoring='neg_mean_absolute_percentage_error')
gs.fit(features, target)

DT_best_score = gs.best_score_
print(" Decision Tree:", DT_best_score) ### Decision Tree: -0.4847991767838546
DT_best_params = gs.best_params_ 
print("Decision Tree:", DT_best_params)###Decision Tree: {'metric': 'manhattan', 'n_neighbors': 4}
###RF
rf_model = RandomForestRegressor(random_state=42) 
rf_params = { 'n_estimators': [50, 100],'max_depth': [None, 5]}
rf_gs = GridSearchCV(rf_model, rf_params, cv=kf, scoring='neg_mean_absolute_percentage_error',n_jobs=-1)
rf_gs.fit(features, target)

rf_best_score = rf_gs.best_score_
print(" Random Forest:", rf_best_score)### Random Forest: -0.18438082363235814
rf_best_params = rf_gs.best_params_ 
print(" Random Forest:", rf_best_params)### Random Forest: {'max_depth': None, 'n_estimators': 100}

###SVR
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
svr_model = SVR() 
svr_params = {'kernel': ['linear','poly', 'rbf'],'C': [0.1, 1,10]}
svr_gs = GridSearchCV(svr_model, svr_params, cv=kf, scoring='neg_mean_absolute_percentage_error',n_jobs=-1)
svr_gs.fit(features_scaled, target)

svr_best_score = svr_gs.best_score_
print("SVR:", svr_best_score)##SVR: -0.22415376340284107
svr_best_params = svr_gs.best_params_ 
print("SVR:", svr_best_params)#SVR: {'C': 10, 'kernel': 'rbf'}


#### بهترین مدل: RF است . پیش بینی های RF به طور متوسط 18.4 % از مقادیر واقعی فاصله دارند . 
## در مدل SVR، پیش بینی ها به طور متوسط 23.71%از مقادیر واقعی فاصله دارند . 
## مدل KNN و درخت تصمیم نیز جواب های یکسانی دادند . 



feature_index = data.feature_names.index("HouseAge")  
plt.scatter(data_New[data.feature_names[feature_index]], data_New['MedHouseValue'], label='Data', alpha=0.5)
plt.title('House Value vs. House Age')
plt.xlabel(data.feature_names[feature_index])
plt.ylabel('Median House Value')
plt.grid()
plt.legend()
plt.show()


house_age_index = data.feature_names.index("HouseAge")
min_age = features[:, house_age_index].min()
max_age = features[:, house_age_index].max()
x_new = np.linspace(min_age, max_age, 100).reshape(-1, 1)
mean_values = np.mean(features, axis=0)
x_new_full = np.tile(mean_values, (x_new.shape[0], 1))
x_new_full[:, house_age_index] = x_new.flatten()

rf_best_model = rf_gs.best_estimator_
y_pred_rf_line = rf_best_model.predict(x_new_full)

# ترسیم نمودار
plt.scatter(features[:, house_age_index], target, alpha=0.5, label='Data')
plt.plot(x_new, y_pred_rf_line, color='red', label='Predicted Line (Random Forest)')
plt.title('House Age vs. Median House Value (Random Forest)')
plt.xlabel('House Age')
plt.ylabel('Median House Value')
plt.grid()
plt.legend()
plt.show()
