#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

rd = pd.read_csv("river_data.csv")
rs = pd.read_csv("river_stations.csv")
wd = pd.read_csv("weather_data.csv")
ws = pd.read_csv("weather_stations.csv")
station_no_river_dusseldorf = "6335050"
station_no_river_cologne = "6335060"
station_no_river_andernach = "6335070"
station_no_river_friedrich = "6335076"
station_no_river_menden = "6335045"
#station_no_river_eitorf = "6335046"
station_no_weather_sollingen = "4741"
station_no_weather_cologne = "2968"

#saving the data size
rd_m, rd_n = rd.shape
rs_m, rs_n = rs.shape
wd_m, wd_n = wd.shape
ws_m, ws_n = ws.shape


# # Read River Data

# In[2]:


rd_c = pd.read_csv('stations/station_' + station_no_river_cologne + '_river_data.csv')
rd_c.date = pd.to_datetime(rd_c.date, format='%Y-%m-%d')
rd_c = rd_c.set_index('date')
columns = ['station_no','year', 'month']
rd_c.drop(columns, inplace=True, axis=1)

rd_d = pd.read_csv('stations/station_' + station_no_river_dusseldorf + '_river_data.csv')
rd_d.date = pd.to_datetime(rd_d.date, format='%Y-%m-%d')
rd_d = rd_d.set_index('date')
columns = ['station_no','year', 'month']
rd_d.drop(columns, inplace=True, axis=1)

rd_a = pd.read_csv('stations/station_' + station_no_river_andernach + '_river_data.csv')
rd_a.date = pd.to_datetime(rd_a.date, format='%Y-%m-%d')
rd_a = rd_a.set_index('date')
columns = ['station_no','year', 'month','delta1', 'delta2', 'delta3', 'water_level' ]
rd_a.drop(columns, inplace=True, axis=1)

rd_f = pd.read_csv('stations/station_' + station_no_river_friedrich + '_river_data.csv')
rd_f.date = pd.to_datetime(rd_f.date, format='%Y-%m-%d')
rd_f = rd_f.set_index('date')
columns = ['station_no','year', 'month','delta1', 'delta2', 'delta3', 'water_level' ]
rd_f.drop(columns, inplace=True, axis=1)

rd_m = pd.read_csv('stations/station_' + station_no_river_menden + '_river_data.csv')
rd_m.date = pd.to_datetime(rd_m.date, format='%Y-%m-%d')
rd_m = rd_m.set_index('date')
columns = ['station_no','year', 'month','delta1', 'delta2', 'delta3', 'water_level' ]
#columns = ['station_no','year', 'month' ]
rd_m.drop(columns, inplace=True, axis=1)


rd_d.delta1.plot()


# Checking the Correlation between water level and discharge at cologne

# In[3]:


corrcoef_water_discharge_cologne = np.corrcoef(rd_c['water_level'], rd_c['discharge'])


# In[4]:


rd_c[1:100000].plot(kind='scatter', x='water_level', y='discharge', color='blue')
plt.title("Rhine River Cologne Station")
plt.show()


# # Read the Weather Data

# In[5]:


wd_c = pd.read_csv('weather_stations/station_' + station_no_weather_cologne + '_weather_data.csv')
wd_c.date = pd.to_datetime(wd_c.date, format='%Y-%m-%d')
wd_c = wd_c.set_index('date')
columns = ['STATIONS_ID','RSF', 'SH_TAG', 'NSH_TAG', 'year', 'month']
wd_c.drop(columns, inplace=True, axis=1)

wd_d = pd.read_csv('weather_stations/station_' + station_no_weather_sollingen + '_weather_data.csv')
wd_d.date = pd.to_datetime(wd_d.date, format='%Y-%m-%d')
wd_d = wd_d.set_index('date')
columns = ['STATIONS_ID','RSF', 'SH_TAG', 'NSH_TAG', 'year', 'month']
wd_d.drop(columns, inplace=True, axis=1)


# In[6]:



combine_df_cd = pd.merge(wd_c, wd_d, how='inner', on=['date'])


# In[7]:


#np.nan_to_num(combine_df_cr, nan=0)
#a = corrcoef_water_discharge_cologne = np.corrcoef(combine_df_cr['RS_x'], combine_df_cr['RS_y'])
a = combine_df_cd[['RS_x', 'RS_y']].corr()
a


# # Combine the weather and river data at cologne

# In[8]:


combine_df_c = pd.merge(rd_c, wd_c, how='left', on=['date'])
combine_df_d = pd.merge(rd_d, wd_d, how='left', on=['date'])
#combine_df_c.to_csv(f'./cologne_combined_data.csv')


# In[9]:


combine_df_c[1090:1095]


# Adjust the combined data

# In[10]:


combine_df_d[1090:1100]


# save the combined data

# In[11]:


#combine_df_c.to_csv(f'./cologne_combined_data.csv')


# In[12]:


#combine_df_c.to_csv(f'./combined_cologne_data.csv')


# # Add other stations

# In[13]:


combine_df_c = pd.merge(combine_df_c, rd_a, how='left', on=['date'])
#combine_df_c = pd.merge(combine_df_c, rd_f, how='left', on=['date'])

columns = ['delta1', 'delta2', 'delta3']
rd_c.drop(columns, inplace=True, axis=1)

combine_df_d = pd.merge(combine_df_d, rd_c, how='left', on=['date'])


# In[14]:


combine_df_d = pd.merge(combine_df_d, rd_a, how='left', on=['date'])
#combine_df_d = pd.merge(combine_df_d, rd_f, how='left', on=['date'])
#combine_df_d = pd.merge(combine_df_d, rd_m, how='left', on=['date'])
combine_df_d.rename(columns={'water_level_x': 'water_level'}, inplace=True)

columns = ['RS']
combine_df_c.drop(columns, inplace=True, axis=1)
combine_df_d.drop(columns, inplace=True, axis=1)


# In[15]:


combine_df_c[100:110]
#combine_df_c.to_csv(f'./combined_cologne_data.csv')


# # Train the model (Functions)

# In[16]:


def define_X_y(df, pred_start, prev_X, prev_y):
    head_df = df[df.index < pred_start]
    #head_df.drop(['station_no', 'STATIONS_ID', 'year_x', 'month_x','RSF', 'SH_TAG', 'NSH_TAG'], axis=1, inplace=True)
    #head_df['RS_y'] = head_df.RS.shift(1)
    #head_df['RS_t'] = head_df.RS.shift(-1)
    #print(head_df)
    
    y_last = head_df['water_level'][-1]
    X = head_df
    y1 = head_df.water_level.shift(-1)
    y2 = head_df.water_level.shift(-2)
    y3 = head_df.water_level.shift(-3)
    X = np.nan_to_num(X, nan=0)
    #scaler = MinMaxScaler()
    X_normalized = X#scaler.fit_transform(X)
    #X_normalized = preprocessing.normalize(X, norm='l2')#X
    x_last = X_normalized[-1,:]
    x_last = x_last.reshape((1, np.shape(x_last)[0]))
    X_normalized = X_normalized[3:-3]
    y1 = y1[3:-3]
    y2 = y2[3:-3]
    y3 = y3[3:-3]
    if prev_X.shape[0] > 3:
        X_normalized = np.concatenate((prev_X, X_normalized))
        y1 = np.concatenate((prev_y[:,0], y1))
        y2 = np.concatenate((prev_y[:,1], y2))
        y3 = np.concatenate((prev_y[:,2], y3))
    return X_normalized, x_last, y_last, y1, y2, y3


# In[17]:


def train_the_model(X, x_last, y_last, y1, y2, y3):
    X_train, X_test, y_train1, y_test1 = train_test_split(X, y1, test_size=0.05, random_state=0)
    regressor1 = LinearRegression()  
    regressor1.fit(X_train, y_train1) #training the algorithm
  
    clf1 = SVR(kernel='rbf', C=100, epsilon=0.01,tol = 0.0001, max_iter = 200)
    clf1.fit(X_train, y_train1)
    y_real_svm1 = clf1.predict(x_last)
    y_real_svm1 = y_real_svm1 - y_last   

    y_pred1 = regressor1.predict(X_test)
    y_pred1 = np.diff(y_pred1)
    y_test1 = np.diff(y_test1)
    y_real1 = regressor1.predict(x_last)
    y_real1 = y_real1 - y_last
    
    X_train, X_test, y_train2, y_test2 = train_test_split(X, y2, test_size=0.05, random_state=5)
    regressor2 = LinearRegression()  
    regressor2.fit(X_train, y_train2) #training the algorithm

    clf2 = SVR(kernel='rbf', C=100, epsilon=0.01,tol = 0.0001, max_iter = 200)
    clf2.fit(X_train, y_train2)
    y_real_svm2 = clf2.predict(x_last)
    y_real_svm2 = y_real_svm2 - y_last  
    
    y_pred2 = regressor2.predict(X_test)
    y_pred2 = np.diff(y_pred2)
    y_test2 = np.diff(y_test2)
    y_real2 = regressor2.predict(x_last)
    y_real2 = y_real2 - y_last

    X_train, X_test, y_train3, y_test3 = train_test_split(X, y3, test_size=0.05, random_state=2)
    regressor3 = LinearRegression()  
    regressor3.fit(X_train, y_train3) #training the algorithm
    
    clf3 = SVR(kernel='rbf', C=100, epsilon=0.01,tol = 0.0001, max_iter = 200)
    clf3.fit(X_train, y_train3)
    y_real_svm3 = clf3.predict(x_last)
    y_real_svm3 = y_real_svm3 - y_last 
    
    y_pred3 = regressor3.predict(X_test)
    y_pred3 = np.diff(y_pred3)
    y_test3 = np.diff(y_test3)
    y_real3 = regressor3.predict(x_last)
    y_real3 = y_real3 - y_last
    
    y_true = np.concatenate((y_test1, y_test2, y_test3), axis=None)
    y_pred = np.concatenate((y_pred1, y_pred2, y_pred3), axis=None)
    y_real = np.concatenate((y_real1, y_real2, y_real3), axis=None)
    y_real = y_real.reshape((np.shape(y_real)[0],1))
    y_real_svm = np.concatenate((y_real_svm1, y_real_svm2, y_real_svm3), axis=None)
    y_real_svm = y_real_svm.reshape((np.shape(y_real_svm)[0],1))
    return y_true, y_pred, y_real, y_real_svm


# # Training Cologne

# In[18]:


skip_missing_days = 16
full_range = pd.date_range(combine_df_c.index.min(), combine_df_c.index.max()+pd.Timedelta(days=1)) #TODO add last day!
combine_df_c = combine_df_c.reindex(full_range, fill_value=np.NaN)

next_gap = pd.to_datetime(combine_df_c.water_level.isnull().idxmax())
tail_gap = next_gap + pd.Timedelta(days=skip_missing_days)
prev_X = pd.DataFrame(np.zeros([1, 9])*0)
prev_y = pd.DataFrame(np.zeros([1, 3])*0)
r2_results = np.zeros(1)
X_real_test_c = np.zeros([1,np.shape(combine_df_c)[1]])
i=0
y_real_test_c = np.zeros([1,1])
y_real_test_svm_c = np.zeros([1,1])


# In[19]:


while next_gap:
    X, x_last, y_last, y1, y2, y3 = define_X_y(combine_df_c, next_gap, prev_X, prev_y)
    X_real_test_c = np.vstack((X_real_test_c, x_last))
    X = np.nan_to_num(X, nan=0)
    indexNames = combine_df_c[ combine_df_c.index < tail_gap ].index
    combine_df_c.drop(indexNames , inplace=True)
    y1 = np.nan_to_num(y1, nan=0)
    y2 = np.nan_to_num(y2, nan=0)
    y3 = np.nan_to_num(y3, nan=0)
    y_true, y_pred, y_real, y_real_svm = train_the_model(X, x_last, y_last, y1,y2,y3)
    y_real_test_c = np.vstack((y_real_test_c, y_real))
    y_real_test_svm_c = np.vstack((y_real_test_svm_c, y_real_svm))
    if np.shape(combine_df_c)[0] < 2:
        break
    next_gap = pd.to_datetime(combine_df_c.water_level.isnull().idxmax())
    tail_gap = next_gap + pd.Timedelta(days=skip_missing_days)
    prev_X = X
    prev_y = [y1, y2, y3]
    prev_y = np.transpose(prev_y)
    i += 1
    print(i)
        #combine_df_c[next_gap < combine_df_c.index & tail_gap > combine_df_c.index]


# In[20]:


X_real_test_c = X_real_test_c[1:-1,:]
y_real_test_c = y_real_test_c[1:-3,:]
y_real_test_svm_c = y_real_test_svm_c[1:-3,:]
r2_score(y_true, y_pred)


# # Training Dusseldorf

# In[21]:


skip_missing_days = 16
full_range = pd.date_range(combine_df_d.index.min(), combine_df_d.index.max()+pd.Timedelta(days=1)) #TODO add last day!
combine_df_d = combine_df_d.reindex(full_range, fill_value=np.NaN)

next_gap = pd.to_datetime(combine_df_d.water_level.isnull().idxmax())
tail_gap = next_gap + pd.Timedelta(days=skip_missing_days)
prev_X = pd.DataFrame(np.zeros([1, 9])*0)
prev_y = pd.DataFrame(np.zeros([1, 3])*0)
r2_results = np.zeros(1)
X_real_test_d = np.zeros([1,np.shape(combine_df_d)[1]])
i=0
y_real_test_d = np.zeros([1,1])
y_real_test_svm_d = np.zeros([1,1])


# In[22]:


while next_gap:
    X, x_last, y_last, y1, y2, y3 = define_X_y(combine_df_d, next_gap, prev_X, prev_y)
    X_real_test_d = np.vstack((X_real_test_d, x_last))
    X = np.nan_to_num(X, nan=0)
    indexNames = combine_df_d[ combine_df_d.index < tail_gap ].index
    combine_df_d.drop(indexNames , inplace=True)
    y1 = np.nan_to_num(y1, nan=0)
    y2 = np.nan_to_num(y2, nan=0)
    y3 = np.nan_to_num(y3, nan=0)
    y_true, y_pred, y_real, y_real_svm = train_the_model(X,x_last, y_last, y1,y2,y3)
    y_real_test_d = np.vstack((y_real_test_d, y_real))
    y_real_test_svm_d = np.vstack((y_real_test_svm_d, y_real_svm))
    if np.shape(combine_df_d)[0] < 2:
        break
    next_gap = pd.to_datetime(combine_df_d.water_level.isnull().idxmax())
    tail_gap = next_gap + pd.Timedelta(days=skip_missing_days)
    prev_X = X
    prev_y = [y1, y2, y3]
    prev_y = np.transpose(prev_y)
    i += 1
    print(i)
        #combine_df_c[next_gap < combine_df_c.index & tail_gap > combine_df_c.index]


# In[23]:


X_real_test_d = X_real_test_d[1:-1,:]
y_real_test_d = y_real_test_d[1:-3,:]
y_real_test_svm_d = y_real_test_svm_d[1:-3,:]
r2_score(y_true, y_pred)


# # Export the results

# In[24]:


y_real_test = np.vstack((y_real_test_d, y_real_test_c))
predict_df = pd.read_csv('./to_predict.csv', index_col=False)
predict_df


# In[25]:


#predict_df.append(y_real_test, ignore_index=True)
#a = pd.DataFrame.from_records(y_real_test)
#predict_df = pd.concat([predict_df, a], axis=1, ignore_index=True)
predict_df['delta'] = y_real_test
#predict_df.reset_index(level='date')
#predict_df = np.concatenate((predict_df, y_real_test), axis=1)
predict_df


# In[26]:


predict_df.to_csv('./team9submission.csv', index=False)
#np.savetxt('./team9submission.csv',predict_df ,delimiter=',')


# # Results for SVM

# In[27]:


y_real_test_svm = np.vstack((y_real_test_svm_d, y_real_test_svm_c))
predict_df = pd.read_csv('./to_predict.csv', index_col=False)
predict_df


# In[28]:


#predict_df.append(y_real_test, ignore_index=True)
#a = pd.DataFrame.from_records(y_real_test)
#predict_df = pd.concat([predict_df, a], axis=1, ignore_index=True)
predict_df['delta'] = y_real_test_svm
#predict_df.reset_index(level='date')
#predict_df = np.concatenate((predict_df, y_real_test), axis=1)
predict_df


# # Calculate R2 Score

# In[29]:


r2_score(y_true, y_pred)


# In[30]:


print(np.shape(prev_X))
print(np.shape(prev_y))
print(np.shape(X))
print(np.shape(y1))
print(np.shape(y_true))
print(np.shape(y_pred))
print(np.shape(X_real_test_d))
print(np.shape(y_real_test_d))
print(np.shape(y_real_test_svm_d))
print(np.shape(y_real_test_c))
print(np.shape(y_real_test_svm_c))
print(np.shape(y_real_test))
print(np.shape(y_real))


# In[31]:


#np.savetxt('./X_real_test_c.csv',X_real_test ,delimiter=',')


# In[32]:


np.shape(x_last)


# In[33]:


y_real_test


# In[34]:


y_real_test_svm


# In[35]:


#predict_df.to_csv('./team9submission.csv', index=False)


# In[36]:


np.mean(y_real_test)


# In[37]:


np.mean(y_real_test_svm)


# In[38]:


predict_mock = pd.read_csv('./team9submission-rev.csv', index_col=False)


# In[39]:


np.mean(predict_mock['delta'])


# In[ ]:




