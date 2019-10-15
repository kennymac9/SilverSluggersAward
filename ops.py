# %load runs.py
#!/usr/bin/env python

# In[ ]:





# In[236]:


import pandas as pd, numpy as np, matplotlib, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[237]:


allData = pd.read_csv('all-np.csv')


# In[238]:


# Some preprocessing on the column names
allData.astype({'Season': 'category'}).dtypes # set season to category instead of int
allData.set_index(['playerid', 'Name'])

# drop NaN values, which only come into play for pitchers running the bases (a very atypical occurrence)
allData = allData.dropna()

# sort all the rows by player ID first, then season. Makes it easier to figure out which player played which seasons
allData = allData.sort_values(by=['playerid', 'Season'])

# scale all relevant stats
allData[['BB%+', 'Hard%', 'LD+%', 'FB%+', 'Soft%', 'PA']] = preprocessing.scale(allData[['BB%+', 'Hard%', 'LD+%', 'FB%+', 'Soft%', 'PA']])

# assign X and Y sets for regression
X = allData[['BB%+', 'Hard%', 'LD+%', 'FB%+', 'Soft%', 'PA']].values
Y = allData[['OPS']].values

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
        if row['Season'] == 2016:
            played1617 = True
            allData.set_value(index, 'weight', 0.3)
        elif row['Season'] == 2017:
            if played1617:
                allData.set_value(index, 'weight', 0.7)
            else:
                allData.set_value(index, 'weight', 0.3)
        else:
            allData.set_value(index, 'weight', 0.7)
    else:
        allData.set_value(index, 'weight', 1)

print(allData.where(allData['playerid'] == 8090).dropna())
# add newly calculated weighted stats to our dataframe as columns
allData['weightBB'] = allData['weight']*allData['BB%+']
allData['weightHH'] = allData['weight']*allData['Hard%']
allData['weightLD'] = allData['weight']*allData['LD+%']
allData['weightFB'] = allData['weight']*allData['FB%+']
allData['weightSH'] = allData['weight']*allData['Soft%']
allData['weightPA'] = allData['weight']*allData['PA']

# sum up weighted averages by player, so that all years are combined
weightedRunStats2019 = allData.groupby('playerid').sum()
weightedRunStats2019 = weightedRunStats2019[['weightBB', 'weightHH', 'weightLD', 'weightFB', 'weightSH', 'weightPA']]

# we lost player names when doing the group by and sum, so get the player names and put it back in our new data frame
names = []
for i in pid:
    names.append(playerNames[i])
weightedRunStats2019['Name'] = names
weightedRunStats2019['Season'] = 2019

# move name and season columns to the front of the dataframe
cols = weightedRunStats2019.columns.tolist()
cols = cols[-2:] + cols[:-2]
weightedRunStats2019 = weightedRunStats2019[cols]


# In[239]:


# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X, Y)


# In[240]:


X_2019 = weightedRunStats2019[['weightBB', 'weightHH', 'weightLD', 'weightFB', 'weightSH', 'weightPA']].values
y_pred = regressor.predict(X_2019)


# In[241]:


y_pred_list = [] # list of y_pred so we can add it to a dataframe
for i in range(len(y_pred)):
    y_pred_list.append(y_pred[i][0])

weightedRunStats2019['opsPredicted'] = y_pred_list
print(weightedRunStats2019.sort_values(by=['opsPredicted'], ascending=False))
# weightedRunStats2019.to_csv("OPSPredictions.csv", encoding='utf-8', index=False)


# In[ ]:




