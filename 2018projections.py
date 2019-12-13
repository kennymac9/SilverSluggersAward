#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd, numpy as np, matplotlib, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
#import seaborn as seabornInstance
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[237]:


allData = pd.read_csv('all-np15-17.csv')


# In[238]:


# Some preprocessing on the column names
allData.astype({'Season': 'category'}).dtypes # set season to category instead of int
allData.set_index(['playerid', 'Name'])

# drop NaN values, which only come into play for pitchers running the bases (a very atypical occurrence)
allData = allData.dropna()

# sort all the rows by player ID first, then season. Makes it easier to figure out which player played which seasons
allData = allData.sort_values(by=['playerid', 'Season'])


# In[3]:


#!/usr/bin/env python
# coding: utf-8

# Runs

# In[236]:


import pandas as pd, numpy as np, matplotlib, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import seaborn as seabornInstance 
from sklearn.svm import LinearSVR
from sklearn import metrics
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[237]:

allData = pd.read_csv('all-np15-17.csv')


# Some preprocessing on the column names
allData.astype({'Season': 'category'}).dtypes # set season to category instead of int
allData.set_index(['playerid', 'Name'])

# drop NaN values, which only come into play for pitchers running the bases (a very atypical occurrence)
allData = allData.dropna()

# dictionary assigning number of years in our data set played to each player
playerYears = dict() # dict of {'player ID': 'years in our data set that they played (1, 2, or 3)'}
pid, yearsPlayed = np.unique(allData['playerid'], return_counts=True)
for i in range(len(pid)):
    playerYears[pid[i]] = yearsPlayed[i]

# dictionary assigning names to player IDs
playerNames = dict() # dict where key is player ID, value is player name
# Assign each player ID to a name for future purposes
for i, r in allData.iterrows():
    playerNames[r['playerid']] = r['Name']

# assigns weights to each player and the data for that year. If played 2016-18, keep the weights of 0.6, 0.3, 0.1. If played two years, 0.7 and 0.3. If only played one year, assign 1 for now
for index, row in allData.iterrows():
    if playerYears[row['playerid']] == 3:
        continue
    elif playerYears[row['playerid']] == 2: # played in 16-17 or 17-18
        played1617 = False
        if row['Season'] == 2015:
            played1617 = True
            allData.set_value(index, 'weight', 0.3)
        elif row['Season'] == 2016:
            if played1617:
                allData.set_value(index, 'weight', 0.7)
            else:
                allData.set_value(index, 'weight', 0.3)
        else:
            allData.set_value(index, 'weight', 0.7)
    else:
        allData.set_value(index, 'weight', 1)

# sort all the rows by player ID first, then season. Makes it easier to figure out which player played which seasons
allData = allData.sort_values(by=['playerid', 'Season'])

# scale all relevant stats
allData[['wRC', 'PA', 'H', 'AB', 'RBI', 'G', '2B']] = preprocessing.scale(allData[['wRC', 'PA', 'H', 'AB', 'RBI', 'G', '2B']])

# assign X and Y sets for regression
X = allData[['wRC', 'PA', 'H', 'AB', 'RBI', 'G', '2B']].values
Y = allData[['R']].values

# add newly calculated weighted stats to our dataframe as columns
allData['weightWRC'] = allData['weight']*allData['wRC']
allData['weightPA'] = allData['weight']*allData['PA']
allData['weightH'] = allData['weight']*allData['H']
allData['weightAB'] = allData['weight']*allData['AB']
allData['weightRBI'] = allData['weight']*allData['RBI']
allData['weightG'] = allData['weight']*allData['G']
allData['weight2B'] = allData['weight']*allData['2B']

# sum up weighted averages by player, so that all years are combined
weightedRunStats2018 = allData.groupby('playerid', as_index=False).sum()
weightedRunStats2018 = weightedRunStats2018[['playerid', 'weightWRC', 'weightPA', 'weightH', 'weightAB', 'weightRBI', 'weightG', 'weight2B']]

# we lost player names when doing the group by and sum, so get the player names and put it back in our new data frame
names = []
for i in pid:
    names.append(playerNames[i])
weightedRunStats2018['Name'] = names
weightedRunStats2018['Season'] = 2018

# move name and season columns to the front of the dataframe
cols = weightedRunStats2018.columns.tolist()
cols = cols[-2:] + cols[:-2]
weightedRunStats2018 = weightedRunStats2018[cols]


# In[239]:


# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X, Y)


# In[240]:


X_2018 = weightedRunStats2018[['weightWRC', 'weightPA', 'weightH', 'weightAB', 'weightRBI', 'weightG', 'weight2B']].values
y_pred = regressor.predict(X_2018)


# In[241]:


y_pred_list = [] # list of y_pred so we can add it to a dataframe
for i in range(len(y_pred)):
    y_pred_list.append(y_pred[i][0])

weightedRunStats2018['linearRegRuns'] = y_pred_list

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regr = LinearSVR(random_state=0)
regr.fit(X, Y)

# In[240]:
y_pred = regr.predict(X_2018)
weightedRunStats2018['linearSVRRuns'] = y_pred

regr = SVR()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedRunStats2018['svrRuns'] = y_pred

regr = linear_model.BayesianRidge()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedRunStats2018['BayesRidgeRuns'] = y_pred

regr = linear_model.HuberRegressor()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedRunStats2018['HuberRegressorRuns'] = y_pred

regr = linear_model.Ridge()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedRunStats2018['RidgeRuns'] = y_pred

regr = linear_model.ARDRegression()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedRunStats2018['ARDRegressionRuns'] = y_pred

weightedRunStats2018 = weightedRunStats2018.sort_values(by=['linearSVRRuns'], ascending=False)
print(weightedRunStats2018)
# weightedRunStats2018.to_csv('RunsPredicted.csv')


# In[11]:


allData = pd.read_csv('all-np15-17.csv')


# In[238]:


# Some preprocessing on the column names
allData.astype({'Season': 'category'}).dtypes # set season to category instead of int
allData.set_index(['playerid', 'Name'])

# drop NaN values, which only come into play for pitchers running the bases (a very atypical occurrence)
allData = allData.dropna()

# dictionary assigning number of years in our data set played to each player
playerYears = dict() # dict of {'player ID': 'years in our data set that they played (1, 2, or 3)'}
pid, yearsPlayed = np.unique(allData['playerid'], return_counts=True)
for i in range(len(pid)):
    playerYears[pid[i]] = yearsPlayed[i]

# dictionary assigning names to player IDs
playerNames = dict() # dict where key is player ID, value is player name
# Assign each player ID to a name for future purposes
for i, r in allData.iterrows():
    playerNames[r['playerid']] = r['Name']

# assigns weights to each player and the data for that year. If played 2016-18, keep the weights of 0.6, 0.3, 0.1. If played two years, 0.7 and 0.3. If only played one year, assign 1 for now
for index, row in allData.iterrows():
    if playerYears[row['playerid']] == 3:
        continue
    elif playerYears[row['playerid']] == 2: # played in 16-17 or 17-18
        played1617 = False
        if row['Season'] == 2015:
            played1617 = True
            allData.set_value(index, 'weight', 0.3)
        elif row['Season'] == 2016:
            if played1617:
                allData.set_value(index, 'weight', 0.7)
            else:
                allData.set_value(index, 'weight', 0.3)
        else:
            allData.set_value(index, 'weight', 0.7)
    else:
        allData.set_value(index, 'weight', 1)

# sort all the rows by player ID first, then season. Makes it easier to figure out which player played which seasons
allData = allData.sort_values(by=['playerid', 'Season'])

# scale relevants stats
allData[['BABIP', 'OBP', 'wOBA', 'OPS', 'wRC+', 'wRAA', '1B']] = preprocessing.scale(allData[['BABIP', 'OBP', 'wOBA', 'OPS', 'wRC+', 'wRAA', '1B']])

# assign X and Y sets for regression
X = allData[['BABIP', 'OBP', 'wOBA', 'OPS', 'wRC+', 'wRAA', '1B']].values
Y = allData[['AVG']].values

# dictionary assigning number of years in our data set played to each player
playerYears = dict() # dict of {'player ID': 'years in our data set that they played (1, 2, or 3)'}
pid, yearsPlayed = np.unique(allData['playerid'], return_counts=True)
for i in range(len(pid)):
    playerYears[pid[i]] = yearsPlayed[i]

# dictionary assigning names to player IDs
playerNames = dict() # dict where key is player ID, value is player name
# Assign each player ID to a name for future purposes
for i, r in allData.iterrows():
    playerNames[r['playerid']] = r['Name']

# add newly calculated weighted stats to our dataframe as columns
allData['weightBABIP'] = allData['weight']*allData['BABIP']
allData['weightOBP'] = allData['weight']*allData['OBP']
allData['weightwOBA'] = allData['weight']*allData['wOBA']
allData['weightOPS'] = allData['weight']*allData['OPS']
allData['weightwRC+'] = allData['weight']*allData['wRC+']
allData['weightwRAA'] = allData['weight']*allData['wRAA']
allData['weight1B'] = allData['weight']*allData['1B']

# sum up weighted averages by player, so that all years are combined
weightedAvgStats2018 = allData.groupby('playerid', as_index=False).sum()
weightedAvgStats2018 = weightedAvgStats2018[['playerid', 'weightBABIP', 'weightOBP', 'weightwOBA', 'weightOPS', 'weightwRC+', 'weightwRAA', 'weight1B']]

# we lost player names when doing the group by and sum, so get the player names and put it back in our new data frame
names = []
for i in pid:
    names.append(playerNames[i])
weightedAvgStats2018['Name'] = names
weightedAvgStats2018['Season'] = 2018

# move name and season columns to the front of the dataframe
cols = weightedAvgStats2018.columns.tolist()
cols = cols[-2:] + cols[:-2]
weightedAvgStats2018 = weightedAvgStats2018[cols]


# In[13]:


# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X, Y)


# In[14]:


X_2018 = weightedAvgStats2018[['weightBABIP', 'weightOBP', 'weightwOBA', 'weightOPS', 'weightwRC+', 'weightwRAA', 'weight1B']].values
y_pred = regressor.predict(X_2018)


# In[15]:


y_pred_list = [] # list of y_pred so we can add it to a dataframe
for i in range(len(y_pred)):
    y_pred_list.append(y_pred[i][0])

weightedAvgStats2018['linearRegAVG'] = y_pred_list

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regr = LinearSVR(random_state=0)
regr.fit(X, Y)

# In[240]:

y_pred = regr.predict(X_2018)
weightedAvgStats2018['linearSvrAVG'] = y_pred

regr = SVR()
regr.fit(X, Y)

# In[240]:

y_pred = regr.predict(X_2018)
weightedAvgStats2018['svrAVG'] = y_pred

regr = linear_model.BayesianRidge()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedAvgStats2018['BayesRidgeAVG'] = y_pred

regr = linear_model.HuberRegressor()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedAvgStats2018['HuberRegressorAVG'] = y_pred

regr = linear_model.Ridge()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedAvgStats2018['RidgeAVG'] = y_pred

regr = linear_model.ARDRegression()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedAvgStats2018['ARDRegressionAVG'] = y_pred

weightedAvgStats2018 = weightedAvgStats2018.sort_values(by=['linearSvrAVG'], ascending=False)
print(weightedAvgStats2018)
# weightedAvgStats2018.to_csv('AVGPredictions.csv')


# In[12]:



allData = pd.read_csv('all-np15-17.csv')


# In[238]:


# Some preprocessing on the column names
allData.astype({'Season': 'category'}).dtypes # set season to category instead of int
allData.set_index(['playerid', 'Name'])

# drop NaN values, which only come into play for pitchers running the bases (a very atypical occurrence)
allData = allData.dropna()

# dictionary assigning number of years in our data set played to each player
playerYears = dict() # dict of {'player ID': 'years in our data set that they played (1, 2, or 3)'}
pid, yearsPlayed = np.unique(allData['playerid'], return_counts=True)
for i in range(len(pid)):
    playerYears[pid[i]] = yearsPlayed[i]

# dictionary assigning names to player IDs
playerNames = dict() # dict where key is player ID, value is player name
# Assign each player ID to a name for future purposes
for i, r in allData.iterrows():
    playerNames[r['playerid']] = r['Name']

# assigns weights to each player and the data for that year. If played 2016-18, keep the weights of 0.6, 0.3, 0.1. If played two years, 0.7 and 0.3. If only played one year, assign 1 for now
for index, row in allData.iterrows():
    if playerYears[row['playerid']] == 3:
        continue
    elif playerYears[row['playerid']] == 2: # played in 16-17 or 17-18
        played1617 = False
        if row['Season'] == 2015:
            played1617 = True
            allData.set_value(index, 'weight', 0.3)
        elif row['Season'] == 2016:
            if played1617:
                allData.set_value(index, 'weight', 0.7)
            else:
                allData.set_value(index, 'weight', 0.3)
        else:
            allData.set_value(index, 'weight', 0.7)
    else:
        allData.set_value(index, 'weight', 1)


# sort all the rows by player ID first, then season. Makes it easier to figure out which player played which seasons
allData = allData.sort_values(by=['playerid', 'Season'])

# scale relevants stats
allData[['RBI', 'ISO', 'wRC', 'SLG', 'R', 'HR/FB', 'OPS']] = preprocessing.scale(allData[['RBI', 'ISO', 'wRC', 'SLG', 'R', 'HR/FB', 'OPS']])

# assign X and Y sets for regression
X = allData[['RBI', 'ISO', 'wRC', 'SLG', 'R', 'HR/FB', 'OPS']].values
Y = allData[['HR']].values

# add newly calculated weighted stats to our dataframe as columns
allData['weightRBI'] = allData['weight']*allData['RBI']
allData['weightISO'] = allData['weight']*allData['ISO']
allData['weightwRC'] = allData['weight']*allData['wRC']
allData['weightSLG'] = allData['weight']*allData['SLG']
allData['weightR'] = allData['weight']*allData['R']
allData['weightHRFB'] = allData['weight']*allData['HR/FB']
allData['weightOPS'] = allData['weight']*allData['OPS']

# sum up weighted averages by player, so that all years are combined
weightedHRStats2018 = allData.groupby('playerid', as_index=False).sum()
weightedHRStats2018 = weightedHRStats2018[['playerid', 'weightRBI', 'weightISO', 'weightwRC', 'weightSLG', 'weightR', 'weightHRFB', 'weightOPS']]

# we lost player names when doing the group by and sum, so get the player names and put it back in our new data frame
names = []
for i in pid:
    names.append(playerNames[i])
weightedHRStats2018['Name'] = names
weightedHRStats2018['Season'] = 2018

# move name and season columns to the front of the dataframe
cols = weightedHRStats2018.columns.tolist()
cols = cols[-2:] + cols[:-2]
weightedHRStats2018 = weightedHRStats2018[cols]


# In[13]:


# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X, Y)


# In[14]:


X_2018 = weightedHRStats2018[['weightRBI', 'weightISO', 'weightwRC', 'weightSLG', 'weightR', 'weightHRFB', 'weightOPS']].values
y_pred = regressor.predict(X_2018)


# In[15]:


y_pred_list = [] # list of y_pred so we can add it to a dataframe
for i in range(len(y_pred)):
    y_pred_list.append(y_pred[i][0])

weightedHRStats2018['linearRegHR'] = y_pred_list

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regr = LinearSVR(random_state=0)
regr.fit(X, Y)

# In[240]:

y_pred = regr.predict(X_2018)
weightedHRStats2018['linearSvrHR'] = y_pred

regr = SVR()
regr.fit(X, Y)

# In[240]:

y_pred = regr.predict(X_2018)
weightedHRStats2018['SvrHR'] = y_pred

regr = linear_model.BayesianRidge()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedHRStats2018['BayesRidgeHR'] = y_pred

regr = linear_model.HuberRegressor()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedHRStats2018['HuberRegressorHR'] = y_pred

regr = linear_model.Ridge()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedHRStats2018['RidgeHR'] = y_pred

regr = linear_model.ARDRegression()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedHRStats2018['ARDRegressionHR'] = y_pred

weightedHRStats2018 = weightedHRStats2018.sort_values(by=['linearSvrHR'], ascending=False)
print(weightedHRStats2018)
# weightedHRStats2018.to_csv('HRPredictions.csv')


# In[13]:



allData = pd.read_csv('all-np15-17.csv')


# In[238]:


# Some preprocessing on the column names
allData.astype({'Season': 'category'}).dtypes # set season to category instead of int
allData.set_index(['playerid', 'Name'])

# drop NaN values, which only come into play for pitchers running the bases (a very atypical occurrence)
allData = allData.dropna()

# dictionary assigning number of years in our data set played to each player
playerYears = dict() # dict of {'player ID': 'years in our data set that they played (1, 2, or 3)'}
pid, yearsPlayed = np.unique(allData['playerid'], return_counts=True)
for i in range(len(pid)):
    playerYears[pid[i]] = yearsPlayed[i]

# dictionary assigning names to player IDs
playerNames = dict() # dict where key is player ID, value is player name
# Assign each player ID to a name for future purposes
for i, r in allData.iterrows():
    playerNames[r['playerid']] = r['Name']

# assigns weights to each player and the data for that year. If played 2016-18, keep the weights of 0.6, 0.3, 0.1. If played two years, 0.7 and 0.3. If only played one year, assign 1 for now
for index, row in allData.iterrows():
    if playerYears[row['playerid']] == 3:
        continue
    elif playerYears[row['playerid']] == 2: # played in 16-17 or 17-18
        played1617 = False
        if row['Season'] == 2015:
            played1617 = True
            allData.set_value(index, 'weight', 0.3)
        elif row['Season'] == 2016:
            if played1617:
                allData.set_value(index, 'weight', 0.7)
            else:
                allData.set_value(index, 'weight', 0.3)
        else:
            allData.set_value(index, 'weight', 0.7)
    else:
        allData.set_value(index, 'weight', 1)


# sort all the rows by player ID first, then season. Makes it easier to figure out which player played which seasons
allData = allData.sort_values(by=['playerid', 'Season'])

# scale all relevant stats
allData[['wOBA', 'wRC+', 'SLG', 'wRAA', 'OBP', 'ISO', 'HR']] = preprocessing.scale(allData[['wOBA', 'wRC+', 'SLG', 'wRAA', 'OBP', 'ISO', 'HR']])

# assign X and Y sets for regression
X = allData[['wOBA', 'wRC+', 'SLG', 'wRAA', 'OBP', 'ISO', 'HR']].values
Y = allData[['OPS']].values


# add newly calculated weighted stats to our dataframe as columns
allData['weightwOBA'] = allData['weight']*allData['wOBA']
allData['weightwRC+'] = allData['weight']*allData['wRC+']
allData['weightSLG'] = allData['weight']*allData['SLG']
allData['weightwRAA'] = allData['weight']*allData['wRAA']
allData['weightOBP'] = allData['weight']*allData['OBP']
allData['weightISO'] = allData['weight']*allData['ISO']
allData['weightHR'] = allData['weight']*allData['HR']

# sum up weighted averages by player, so that all years are combined
weightedOPSStats2018 = allData.groupby('playerid', as_index=False).sum()
weightedOPSStats2018 = weightedOPSStats2018[['playerid', 'weightwOBA', 'weightwRC+', 'weightSLG', 'weightwRAA', 'weightOBP', 'weightISO', 'weightHR']]

# we lost player names when doing the group by and sum, so get the player names and put it back in our new data frame
names = []
for i in pid:
    names.append(playerNames[i])
weightedOPSStats2018['Name'] = names
weightedOPSStats2018['Season'] = 2018

# move name and season columns to the front of the dataframe
cols = weightedOPSStats2018.columns.tolist()
cols = cols[-2:] + cols[:-2]
weightedOPSStats2018 = weightedOPSStats2018[cols]


# In[239]:


# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X, Y)


# In[240]:


X_2018 = weightedOPSStats2018[['weightwOBA', 'weightwRC+', 'weightSLG', 'weightwRAA', 'weightOBP', 'weightISO', 'weightHR']].values
y_pred = regressor.predict(X_2018)


# In[241]:


y_pred_list = [] # list of y_pred so we can add it to a dataframe
for i in range(len(y_pred)):
    y_pred_list.append(y_pred[i][0])

weightedOPSStats2018['linearRegOPS'] = y_pred_list

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regr = LinearSVR(random_state=0)
regr.fit(X, Y)

# In[240]:

y_pred = regr.predict(X_2018)
weightedOPSStats2018['linearSvrOPS'] = y_pred

regr = SVR()
regr.fit(X, Y)

# In[240]:

y_pred = regr.predict(X_2018)
weightedOPSStats2018['svrOPS'] = y_pred

regr = linear_model.BayesianRidge()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedOPSStats2018['BayesRidgeOPS'] = y_pred

regr = linear_model.HuberRegressor()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedOPSStats2018['HuberRegressorOPS'] = y_pred

regr = linear_model.Ridge()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedOPSStats2018['RidgeOPS'] = y_pred

regr = linear_model.ARDRegression()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedOPSStats2018['ARDRegressionOPS'] = y_pred


weightedOPSStats2018 = weightedOPSStats2018.sort_values(by=['linearSvrOPS'], ascending=False)
print(weightedOPSStats2018)
# weightedOPSStats2018.to_csv('OPSPredictions.csv')


# In[14]:



allData = pd.read_csv('all-np15-17.csv')


# In[238]:


# Some preprocessing on the column names
allData.astype({'Season': 'category'}).dtypes # set season to category instead of int
allData.set_index(['playerid', 'Name'])

# drop NaN values, which only come into play for pitchers running the bases (a very atypical occurrence)
allData = allData.dropna()

# dictionary assigning number of years in our data set played to each player
playerYears = dict() # dict of {'player ID': 'years in our data set that they played (1, 2, or 3)'}
pid, yearsPlayed = np.unique(allData['playerid'], return_counts=True)
for i in range(len(pid)):
    playerYears[pid[i]] = yearsPlayed[i]

# dictionary assigning names to player IDs
playerNames = dict() # dict where key is player ID, value is player name
# Assign each player ID to a name for future purposes
for i, r in allData.iterrows():
    playerNames[r['playerid']] = r['Name']

# assigns weights to each player and the data for that year. If played 2016-18, keep the weights of 0.6, 0.3, 0.1. If played two years, 0.7 and 0.3. If only played one year, assign 1 for now
for index, row in allData.iterrows():
    if playerYears[row['playerid']] == 3:
        continue
    elif playerYears[row['playerid']] == 2: # played in 16-17 or 17-18
        played1617 = False
        if row['Season'] == 2015:
            played1617 = True
            allData.set_value(index, 'weight', 0.3)
        elif row['Season'] == 2016:
            if played1617:
                allData.set_value(index, 'weight', 0.7)
            else:
                allData.set_value(index, 'weight', 0.3)
        else:
            allData.set_value(index, 'weight', 0.7)
    else:
        allData.set_value(index, 'weight', 1)


# sort all the rows by player ID first, then season. Makes it easier to figure out which player played which seasons
allData = allData.sort_values(by=['playerid', 'Season'])

# scale relevants stats
allData[['wRC', 'HR',  'PA', 'AB', 'R', 'H', '2B']] = preprocessing.scale(allData[['wRC', 'HR',  'PA', 'AB', 'R', 'H', '2B']])

# assign X and Y sets for regression
X = allData[['wRC', 'HR',  'PA', 'AB', 'R', 'H', '2B']].values
Y = allData[['RBI']].values

# add newly calculated weighted stats to our dataframe as columns
allData['weightwRC'] = allData['weight']*allData['wRC']
allData['weightHR'] = allData['weight']*allData['HR']
allData['weightPA'] = allData['weight']*allData['PA']
allData['weightAB'] = allData['weight']*allData['AB']
allData['weightR'] = allData['weight']*allData['R']
allData['weightH'] = allData['weight']*allData['H']
allData['weight2B'] = allData['weight']*allData['2B']


# sum up weighted averages by player, so that all years are combined
weightedRBIStats2018 = allData.groupby('playerid', as_index=False).sum()
weightedRBIStats2018 = weightedRBIStats2018[['playerid', 'weightwRC', 'weightHR', 'weightPA', 'weightAB', 'weightR' ,'weightH', 'weight2B']]

# we lost player names when doing the group by and sum, so get the player names and put it back in our new data frame
names = []
for i in pid:
    names.append(playerNames[i])
weightedRBIStats2018['Name'] = names
weightedRBIStats2018['Season'] = 2018

# move name and season columns to the front of the dataframe
cols = weightedRBIStats2018.columns.tolist()
cols = cols[-2:] + cols[:-2]
weightedRBIStats2018 = weightedRBIStats2018[cols]


# In[13]:


# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X, Y)


# In[14]:


X_2018 = weightedRBIStats2018[['weightwRC', 'weightHR', 'weightPA', 'weightAB', 'weightR' ,'weightH', 'weight2B']].values
y_pred = regressor.predict(X_2018)


# In[15]:


y_pred_list = [] # list of y_pred so we can add it to a dataframe
for i in range(len(y_pred)):
    y_pred_list.append(y_pred[i][0])

weightedRBIStats2018['linearRegRBI'] = y_pred_list

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regr = LinearSVR(random_state=0)
regr.fit(X, Y)

# In[240]:

y_pred = regr.predict(X_2018)
weightedRBIStats2018['linearSvrRBI'] = y_pred

regr = SVR()
regr.fit(X, Y)

# In[240]:

y_pred = regr.predict(X_2018)
weightedRBIStats2018['svrRBI'] = y_pred

regr = linear_model.BayesianRidge()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedRBIStats2018['BayesRidgeRBI'] = y_pred

regr = linear_model.HuberRegressor()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedRBIStats2018['HuberRegressorRBI'] = y_pred

regr = linear_model.Ridge()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedRBIStats2018['RidgeRBI'] = y_pred

regr = linear_model.ARDRegression()
regr.fit(X, Y)

y_pred = regr.predict(X_2018)
weightedRBIStats2018['ARDRegressionRBI'] = y_pred

weightedRBIStats2018 = weightedRBIStats2018.sort_values(by=['linearSvrRBI'], ascending=False)
print(weightedRBIStats2018)
# weightedRBIStats2018.to_csv('RBIPredictions.csv')


# In[15]:


weightedRBIStats2018.reset_index()
dfa = weightedRBIStats2018.join(weightedAvgStats2018, how='outer', lsuffix='playerid', rsuffix='playerid')
dfa = dfa.join(weightedHRStats2018, how='outer', lsuffix='playerid', rsuffix='playerid')
dfa = dfa.join(weightedOPSStats2018, how='outer', lsuffix='playerid', rsuffix='playerid')
dfa = dfa.join(weightedRunStats2018, how='outer', lsuffix='playerid', rsuffix='playerid')
print(dfa)
dfa.to_csv('allPredictions2018.csv')


# In[ ]:




