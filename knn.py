#!/usr/bin/env python
# coding: utf-8

# In[437]:


import pandas as pd, numpy as np, matplotlib, matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
import math

initYear = '2016'
years = [initYear, str(int(initYear) + 1), str(int(initYear) + 2)]
lastYear = int(years[2])


# use this read csv for 2017-2019 predictions
# allData = pd.read_csv(str(lastYear + 1) + 'PredictionsAndActual.csv').dropna()

# use this for kenny's stuff
allData = pd.read_csv('allPredictions2019.csv').dropna()
# allData = pd.read_csv(str(lastYear + 1) + 'PredictionsAndActual.csv')
# data = pd.read_csv('allPredictions2019.csv')
# data = data[['playerid', 'HuberRegressorRBI', 'HuberRegressorAVG', 'HuberRegressorHR', 'HuberRegressorOPS', 'HuberRegressorRuns', 'RidgeRBI', 'RidgeAVG', 'RidgeHR', 'RidgeOPS', 'RidgeRuns']]
# data = data.join(allData, how='outer', lsuffix='playerid', rsuffix='playerid')
# # data = data.rename(columns={'playeridplayerid': 'playerid'})#, 'HRsPredicted': 'HR', 'RBIsPredicted': 'RBI', 'AVGPredicted': 'AVG', 'opsPredicted': 'OPS' })
# allData = data
# allData = allData.dropna()
# print(allData['playeridplayerid'])


# print(list(allData.columns))
# print(allData['playerid'])
# use this read csv if you want to see what will happen in 2020
# allData = pd.read_csv('2020Predictions.csv').dropna()

# use this read csv is you want to see how our old predictions fare
# allData = pd.read_csv('oldPredictions2019.csv').dropna()

def computeData(toScale, colList, dataMode):
    if dataMode == 'scale':
        return preprocessing.scale(toScale[colList])
    else:
        return toScale[colList]


# In[438]:


def getDataByLAP(fileName, mode, dataMode):
    positionData = pd.read_csv(fileName)
    position = positionData.loc[positionData['Season'] == lastYear]
    position = position[['playerid', 'Team']].copy()
    position = pd.merge(position, allData, on='playerid').dropna()
    
    if mode == 'predicted':
        colList = ['runsPredicted', 'HRsPredicted', 'RBIsPredicted', 'AVGPredicted', 'opsPredicted']
    elif mode == 'linearSVR':
        colList = ['linearSVRRuns', 'linearSvrHR', 'linearSvrRBI', 'linearSvrAVG', 'linearSvrOPS']
    elif mode == 'svr':
        colList = ['svrRuns', 'SvrHR', 'svrRBI', 'svrAVG', 'svrOPS']
    elif mode == 'huber':
        colList = ['HuberRegressorRBI', 'HuberRegressorAVG', 'HuberRegressorHR', 'HuberRegressorOPS', 'HuberRegressorRuns']
    elif mode == 'ridge':
        colList = ['RidgeRBI', 'RidgeAVG', 'RidgeHR', 'RidgeOPS', 'RidgeRuns']
    else:
        if lastYear == 2018 or lastYear == 2019:
            colList = ['runsActual', 'HRsActual', 'RBIsActuals', 'AVGActuals', 'OpsActuals']
        else:
            colList = ['runs', 'hr', 'rbi', 'avg', 'ops']
            
    position[colList] = computeData(position, colList, dataMode)

    return position


# In[439]:


def getMin(wins):
    minVal = 100
    minRow = []
    for index, val in enumerate(wins):
        if val['Distance'] < minVal:
            minRow = val
    print(minRow['Name'])
    return minRow


# In[440]:


def compute_by_position(dataMode):
    for league in winners:
        for position in winners[league]:
            if position == 'of':
                continue
            fileName = winners[league][position]['file']

            for year in years: #winners[league][position]['stats']:
                data = pd.read_csv(fileName)
                data = data.loc[data['Season'] == int(year)]
                colList = ['R', 'HR', 'RBI', 'AVG', 'OPS']
                data[colList] = computeData(data, colList, dataMode)

                if position != 'ofs':
                    data['Award'] = np.where(data['Name'] == winners[league][position]['stats'][year]['name'], 1, 0)
                    winners[league][position]['stats'][year]['playerdata'] = data[['Season', 'Name', 'playerid', 'R', 'HR', 'RBI', 'AVG', 'OPS', 'Award']]
                    data = data.loc[data['Name'] == winners[league][position]['stats'][year]['name']]
                    winner = data.loc[data['Name'] == winners[league][position]['stats'][year]['name']]
                    winner = winner[['R', 'HR', 'RBI', 'AVG', 'OPS']].copy()
                    winners[league][position]['stats'][year]['data'] = winner
                else:
                    for index, row in data.iterrows():
                        if row['Name'] in winners[league][position]['stats'][year]['names']:
                            data.at[index, 'Award'] = 1
                        else:
                            data.at[index, 'Award'] = 0
                    winners[league][position]['stats'][year]['playerdata'] = data[['Season', 'Name', 'playerid', 'R', 'HR', 'RBI', 'AVG', 'OPS', 'Award']]
                    winner = data.loc[data['Award'] == 1]
                    winner = winner[['R', 'HR', 'RBI', 'AVG', 'OPS']].copy()
                    winners[league][position]['stats'][year]['data'] = winner
            winners[league][position]['average'] = pd.concat((winners[league][position]['stats'][years[0]]['data'], winners[league][position]['stats'][years[1]]['data']))
            winners[league][position]['average'] = pd.concat((winners[league][position]['average'], winners[league][position]['stats'][years[2]]['data']))
            winners[league][position]['average'] = winners[league][position]['average'].mean(axis=0)
            winners[league][position]['average'] = pd.DataFrame(data=winners[league][position]['average']).T
            frames = [winners[league][position]['stats'][years[0]]['playerdata'], winners[league][position]['stats'][years[1]]['playerdata'], winners[league][position]['stats'][years[2]]['playerdata']]
            winners[league][position]['playerdata'] = pd.concat(frames)
        winners[league]['of'] = {}
        of1 = pd.DataFrame(data=winners[league]['of1']['average'])
        of2 = pd.DataFrame(data=winners[league]['of2']['average'])
        of3 = pd.DataFrame(data=winners[league]['of3']['average'])
        winners[league]['of']['average'] = pd.concat((of1, of2))
        winners[league]['of']['average'] = pd.concat((winners[league]['of']['average'], of3))
        winners[league]['of']['average'] = winners[league]['of']['average'].mean(axis=0)
        winners[league]['of']['average'] = pd.DataFrame(data=winners[league]['of']['average']).T


# In[441]:


def knn(mode, neighbors, debug):
    for league in winners:
        otherLeague = 'al'
        if league == 'al':
            otherLeague = 'nl'
        predWinners[league] = {}
        for position in winners[league]:
            frames = []
            temp = []
            if position == 'of' or position == 'of1' or position == 'of2' or position == 'of3':
                continue

            if position != 'dh':
                if position == 'ofs':
                    for year in years:
                        frames.append(pd.DataFrame(data=winners[otherLeague]['of1']['stats'][year]['data']))
                        frames.append(pd.DataFrame(data=winners[otherLeague]['of2']['stats'][year]['data']))
                        frames.append(pd.DataFrame(data=winners[otherLeague]['of3']['stats'][year]['data']))
                    temp = pd.concat(frames)
                    temp['Award'] = 1
                    train_x = pd.concat((winners[league][position]['playerdata'][['R', 'HR', 'RBI', 'AVG', 'OPS']],temp[['R', 'HR', 'RBI', 'AVG', 'OPS']]), axis=0)
                    train_y = pd.concat((winners[league][position]['playerdata'][['Award']],temp[['Award']]))
                else:
                    for year in years: #winners[otherLeague][position]['stats']:
                        frames.append(pd.DataFrame(data=winners[otherLeague][position]['stats'][year]['data']))
                    temp = pd.concat(frames)
                    temp['Award'] = 1
                    train_x = pd.concat((winners[league][position]['playerdata'][['R', 'HR', 'RBI', 'AVG', 'OPS']],temp[['R', 'HR', 'RBI', 'AVG', 'OPS']]), axis=0)
                    train_y = pd.concat((winners[league][position]['playerdata'][['Award']],temp[['Award']]))
            else:
                train_x = winners[league][position]['playerdata'][['R', 'HR', 'RBI', 'AVG', 'OPS']]
                train_y = winners[league][position]['playerdata'][['Award']]
                
            for i, p in enumerate(specificData[league]):
                numWinners = 0
                awardWinners = []
                
                if specificData[league][i]['position'] == position:
                    if debug:
                        print(position)
                    predWinners[league][position] = []
                    if mode == 'predicted':
                        test = specificData[league][i]['data'][['playerid', 'Name', 'runsPredicted', 'HRsPredicted', 'RBIsPredicted', 'AVGPredicted', 'opsPredicted']]
                        test = test.rename(columns={'runsPredicted': 'R', 'HRsPredicted': 'HR', 'RBIsPredicted': 'RBI', 'AVGPredicted': 'AVG', 'opsPredicted': 'OPS' })
                    elif mode == 'linearSVR':
                        test = specificData[league][i]['data'][['playerid', 'Name', 'linearSVRRuns', 'linearSvrHR', 'linearSvrRBI', 'linearSvrAVG', 'linearSvrOPS']]
                        test = test.rename(columns={'linearSVRRuns': 'R', 'linearSvrHR': 'HR', 'linearSvrRBI': 'RBI', 'linearSvrAVG': 'AVG', 'linearSvrOPS': 'OPS'})
                    elif mode == 'svr':
                        test = specificData[league][i]['data'][['playerid', 'Name', 'svrRuns', 'SvrHR', 'svrRBI', 'svrAVG', 'svrOPS']]
                        test = test.rename(columns={'svrRuns': 'R', 'SvrHR': 'HR', 'svrRBI': 'RBI', 'svrAVG': 'AVG', 'svrOPS': 'OPS'})
                    elif mode == 'huber':
                        test = specificData[league][i]['data'][['playerid', 'Name', 'HuberRegressorRBI', 'HuberRegressorAVG', 'HuberRegressorHR', 'HuberRegressorOPS', 'HuberRegressorRuns']]
                        test = test.rename(columns={'HuberRegressorRBI': 'RBI', 'HuberRegressorAVG': 'AVG', 'HuberRegressorHR': 'HR', 'HuberRegressorOPS': 'OPS', 'HuberRegressorRuns': 'R'})
                    elif mode == 'ridge':
                        test = specificData[league][i]['data'][['playerid', 'Name', 'RidgeRBI', 'RidgeAVG', 'RidgeHR', 'RidgeOPS', 'RidgeRuns']]
                        test = test.rename(columns={'RidgeRBI': 'RBI', 'RidgeAVG': 'AVG', 'RidgeHR': 'HR', 'RidgeOPS': 'OPS', 'RidgeRuns': 'R'})
                    else:
                        if lastYear == 2018:
                            test = specificData[league][i]['data'][['playerid', 'Name', 'runsActual', 'HRsActual', 'RBIsActuals', 'AVGActuals', 'OpsActuals']]
                            test = test.rename(columns={'runsActual': 'R', 'HRsActual': 'HR', 'RBIsActuals': 'RBI', 'AVGActuals': 'AVG', 'OpsActuals': 'OPS'})
                        else:
                            test = specificData[league][i]['data'][['playerid', 'Name', 'runs', 'hr', 'rbi', 'avg', 'ops']]
                            test = test.rename(columns={'runs': 'R', 'hr': 'HR', 'rbi': 'RBI', 'avg': 'AVG', 'ops': 'OPS'})
                    classifier = KNeighborsClassifier(n_neighbors=neighbors)
                    classifier.fit(train_x, train_y.values.ravel())

                    if position != 'dh':
                        combinedWinners = pd.concat([winners[league][position]['average'], winners[otherLeague][position]['average']])
                        combinedWinners = combinedWinners.mean(axis=0)
                    else:
                        combinedWinners = winners[league][position]['average']
                        
                    for index, row in test.iterrows():
                        vals = [row['R'], row['HR'], row['RBI'], row['AVG'], row['OPS']]
                        test.at[index, 'Distance'] = distance.euclidean([vals], combinedWinners)
                    
                    test = test.sort_values(by='Distance')
                    for index, row in test.iterrows():
                        vals = [row['R'], row['HR'], row['RBI'], row['AVG'], row['OPS']]
                        res = classifier.predict([vals]) == [1]                        
                        if res == 1:
                            numWinners = numWinners + 1
                            awardWinners.append(row)
                            
                    if numWinners == 5:
                        wins = pd.concat(awardWinners)
                        if debug:
                            print(wins['Name'])
                        predWinners[league][position].append(wins['Name'])
                    elif numWinners < 5:
                        runs = 5 - numWinners
                        if numWinners > 0:
                            for row in awardWinners:
                                if debug:
                                    print(row['Name'])
                                predWinners[league][position].append(row['Name'])
                                test = test[test['playerid'] != row['playerid']]
                        for index in range(runs):
                            minRow = test[test['Distance']==test['Distance'].min()]
                            if debug:
                                print(minRow['Name'].tolist()[0])
                            predWinners[league][position].append(minRow['Name'].tolist()[0])
                            test = test[test['playerid'] != minRow['playerid'].tolist()[0]]

                    else:
                        for row in awardWinners:
                            if debug:
                                print(row['Name'])
                            predWinners[league][position].append(row['Name'])
                        
                if specificData[league][i]['position'] == 'of' and position == 'ofs':
                    if debug:
                        print(position)
                    predWinners[league][position] = []

                    if mode == 'predicted':
                        test = specificData[league][i]['data'][['playerid', 'Name', 'runsPredicted', 'HRsPredicted', 'RBIsPredicted', 'AVGPredicted', 'opsPredicted']]
                        test = test.rename(columns={'runsPredicted': 'R', 'HRsPredicted': 'HR', 'RBIsPredicted': 'RBI', 'AVGPredicted': 'AVG', 'opsPredicted': 'OPS' })
                    elif mode == 'linearSVR':
                        test = specificData[league][i]['data'][['playerid', 'Name', 'linearSVRRuns', 'linearSvrHR', 'linearSvrRBI', 'linearSvrAVG', 'linearSvrOPS']]
                        test = test.rename(columns={'linearSVRRuns': 'R', 'linearSvrHR': 'HR', 'linearSvrRBI': 'RBI', 'linearSvrAVG': 'AVG', 'linearSvrOPS': 'OPS'})
                    elif mode == 'svr':
                        test = specificData[league][i]['data'][['playerid', 'Name', 'svrRuns', 'SvrHR', 'svrRBI', 'svrAVG', 'svrOPS']]
                        test = test.rename(columns={'svrRuns': 'R', 'SvrHR': 'HR', 'svrRBI': 'RBI', 'svrAVG': 'AVG', 'svrOPS': 'OPS'})
                    elif mode == 'huber':
                        test = specificData[league][i]['data'][['playerid', 'Name', 'HuberRegressorRBI', 'HuberRegressorAVG', 'HuberRegressorHR', 'HuberRegressorOPS', 'HuberRegressorRuns']]
                        test = test.rename(columns={'HuberRegressorRBI': 'RBI', 'HuberRegressorAVG': 'AVG', 'HuberRegressorHR': 'HR', 'HuberRegressorOPS': 'OPS', 'HuberRegressorRuns': 'R'})
                    elif mode == 'ridge':
                        test = specificData[league][i]['data'][['playerid', 'Name', 'RidgeRBI', 'RidgeAVG', 'RidgeHR', 'RidgeOPS', 'RidgeRuns']]
                        test = test.rename(columns={'RidgeRBI': 'RBI', 'RidgeAVG': 'AVG', 'RidgeHR': 'HR', 'RidgeOPS': 'OPS', 'RidgeRuns': 'R'})

                    else:
                        if lastYear == 2018:
                            test = specificData[league][i]['data'][['playerid', 'Name', 'runsActual', 'HRsActual', 'RBIsActuals', 'AVGActuals', 'OpsActuals']]
                            test = test.rename(columns={'runsActual': 'R', 'HRsActual': 'HR', 'RBIsActuals': 'RBI', 'AVGActuals': 'AVG', 'OpsActuals': 'OPS'})
                        else:
                            test = specificData[league][i]['data'][['playerid', 'Name', 'runs', 'hr', 'rbi', 'avg', 'ops']]
                            test = test.rename(columns={'runs': 'R', 'hr': 'HR', 'rbi': 'RBI', 'avg': 'AVG', 'ops': 'OPS'})

                    classifier = KNeighborsClassifier(n_neighbors=neighbors)
                    classifier.fit(train_x, train_y.values.ravel())

                    if position != 'dh':
                        combinedWinners = pd.concat([winners[league][position]['average'], winners[otherLeague][position]['average']])
                        combinedWinners = combinedWinners.mean(axis=0)
                    else:
                        combinedWinners = winners[league][position]['average']
                        
                    for index, row in test.iterrows():
                        vals = [row['R'], row['HR'], row['RBI'], row['AVG'], row['OPS']]
                        test.at[index, 'Distance'] = distance.euclidean([vals], combinedWinners)

                    for index, row in test.iterrows():
                        vals = [row['R'], row['HR'], row['RBI'], row['AVG'], row['OPS']]
                        res = classifier.predict([vals]) == [1]
                        if res == 1:
                            numWinners = numWinners + 1
                            awardWinners.append(row)

                    if numWinners == 15:
                        wins = pd.concat(awardWinners)
                        if debug:
                            print(wins['Name'])
                        predWinners[league][position].append(wins['Name'])
                    elif numWinners > 15:
                        for row in awardWinners:
                            if debug:
                                print(row['Name'])
                            predWinners[league][position].append(wins['Name'])
                    else:
                        runs = 15 - numWinners
                        if numWinners > 0:
                            for row in awardWinners:
                                if debug:
                                    print(row['Name'])
                                predWinners[league][position].append(row['Name'])
                                test = test[test['playerid'] != row['playerid']]
                        for index in range(runs):
                            minRow = test[test['Distance']==test['Distance'].min()]
                            if debug:
                                print(minRow['Name'].tolist()[0])
                            predWinners[league][position].append(minRow['Name'].tolist()[0])
                            test = test[test['playerid'] != minRow['playerid'].tolist()[0]]


# In[442]:


specificData = {
    'al': [
        {'position': '1b', 'fileName': 'al-1b+2014.csv', 'data': []},
        {'position': '2b', 'fileName': 'al-2b+2014.csv', 'data': []},
        {'position': '3b', 'fileName': 'al-3b+2014.csv', 'data': []},
        {'position': 'ss', 'fileName': 'al-ss+2014.csv', 'data': []},
        {'position': 'of', 'fileName': 'al-of+2014.csv', 'data': []},
        {'position': 'dh', 'fileName': 'al-dh+2014.csv', 'data': []},
        {'position': 'c', 'fileName': 'al-c+2014.csv', 'data': []},
    ],
    'nl': [
        {'position': '1b', 'fileName': 'nl-1b+2014.csv', 'data': []},
        {'position': '2b', 'fileName': 'nl-2b+2014.csv', 'data': []},
        {'position': '3b', 'fileName': 'nl-3b+2014.csv', 'data': []},
        {'position': 'ss', 'fileName': 'nl-ss+2014.csv', 'data': []},
        {'position': 'of', 'fileName': 'nl-of+2014.csv', 'data': []},
        {'position': 'c', 'fileName': 'nl-c+2014.csv', 'data': []},
    ]
}


# In[443]:


winners = {
    'al': {
        '1b': {
            'file': 'al-1b+2014.csv',
            'stats': {
                '2014': { 'name': 'Jose Abreu', 'data': [] },
                '2015': { 'name': 'Miguel Cabrera', 'data': [] },
                '2016': { 'name': 'Miguel Cabrera', 'data': [] },
                '2017': { 'name': 'Eric Hosmer', 'data': [] },
                '2018': { 'name': 'Jose Abreu', 'data': [] },
                '2019': { 'name': 'Carlos Santana', 'data': [] }
            }
        },
        '2b': {
            'file': 'al-2b+2014.csv',
            'stats': {
                '2014': { 'name': 'Jose Altuve', 'data': [] },
                '2015': { 'name': 'Jose Altuve', 'data': [] },
                '2016': { 'name': 'Jose Altuve', 'data': [] },
                '2017': { 'name': 'Jose Altuve', 'data': [] },
                '2018': { 'name': 'Jose Altuve', 'data': [] },
                '2019': { 'name': 'DJ LeMahieu', 'data': [] }
            }
        },
        '3b': {
            'file': 'al-3b+2014.csv',
            'stats': {
                '2014': { 'name': 'Adrian Beltre', 'data': [] },
                '2015': { 'name': 'Josh Donaldson', 'data': [] },
                '2016': { 'name': 'Josh Donaldson', 'data': [] },
                '2017': { 'name': 'Jose Ramirez', 'data': [] },
                '2018': { 'name': 'Jose Ramirez', 'data': [] },
                '2019': { 'name': 'Alex Bregman', 'data': [] }
            }
        },
        'ss': {
            'file': 'al-ss+2014.csv',
            'stats': {
                '2014': { 'name': 'Alexei Ramirez', 'data': [] },
                '2015': { 'name': 'Xander Bogaerts', 'data': [] },
                '2016': { 'name': 'Xander Bogaerts', 'data': [] },
                '2017': { 'name': 'Francisco Lindor', 'data': [] },
                '2018': { 'name': 'Francisco Lindor', 'data': [] },
                '2019': { 'name': 'Xander Bogaerts', 'data': [] }

            }
        },
        'ofs': {
            'file': 'al-of+2014.csv',
            'stats': {
                '2014': {
                    'names': ['Jose Bautista', 'Mike Trout', 'Michael Brantley'],
                    'data': []
                },
                '2015': {
                    'names': ['Nelson Cruz', 'Mike Trout', 'J.D. Martinez'],
                    'data': []
                },
                '2016': {
                    'names': ['Mookie Betts', 'Mike Trout', 'Mark Trumbo'],
                    'data': []
                },
                '2017': {
                    'names': ['Aaron Judge', 'George Springer', 'Justin Upton'],
                    'data': []
                },
                '2018': {
                    'names': ['Mookie Betts', 'Mike Trout', 'J.D. Martinez'],
                    'data': []
                },
                '2019': {
                    'names': ['Mookie Betts', 'Mike Trout', 'George Springer'],
                    'data': []
                }
            }
        },
        'of1': {
            'file': 'al-of+2014.csv',
            'stats': {
                '2014': { 'name': 'Jose Bautista', 'data': [] },
                '2015': { 'name': 'Nelson Cruz', 'data': [] },
                '2016': { 'name': 'Mookie Betts', 'data': [] },
                '2017': { 'name': 'Aaron Judge', 'data': [] },
                '2018': { 'name': 'Mookie Betts', 'data': [] },
                '2019': { 'name': 'Mookie Betts', 'data': [] }
            }
        },
        'of2': {
            'file': 'al-of+2014.csv',
            'stats': {
                '2014': { 'name': 'Mike Trout', 'data': [] },
                '2015': { 'name': 'Mike Trout', 'data': [] },
                '2016': { 'name': 'Mike Trout', 'data': [] },
                '2017': { 'name': 'George Springer', 'data': [] },
                '2018': { 'name': 'Mike Trout', 'data': [] },
                '2019': { 'name': 'Mike Trout', 'data': [] }
            }
        },
        'of3': {
            'file': 'al-of+2014.csv',
            'stats': {
                '2014': { 'name': 'Michael Brantley', 'data': [] },
                '2015': { 'name': 'J.D. Martinez', 'data': [] },
                '2016': { 'name': 'Mark Trumbo', 'data': [] },
                '2017': { 'name': 'Justin Upton', 'data': [] },
                '2018': { 'name': 'J.D. Martinez', 'data': [] },
                '2019': { 'name': 'George Springer', 'data': [] }

            }
        },
        'c': {
            'file': 'al-c+2014.csv',
            'stats': {
                '2014': { 'name': 'Yan Gomes', 'data': [] },
                '2015': { 'name': 'Brian McCann', 'data': [] },
                '2016': { 'name': 'Salvador Perez', 'data': [] },
                '2017': { 'name': 'Gary Sanchez', 'data': [] },
                '2018': { 'name': 'Salvador Perez', 'data': [] },
                '2019': { 'name': 'Mitch Garver', 'data': [] }
            }
        },
        'dh': {
            'file': 'al-dh+2014.csv',
            'stats': {
                '2014': { 'name': 'Victor Martinez', 'data': [] },
                '2015': { 'name': 'Kendrys Morales', 'data': [] },
                '2016': { 'name': 'David Ortiz', 'data': [] },
                '2017': { 'name': 'Nelson Cruz', 'data': [] },
                '2018': { 'name': 'J.D. Martinez', 'data': [] },
                '2019': { 'name': 'Nelson Cruz', 'data': [] }
            }
        }
    },
    'nl': {
        '1b': {
            'file': 'nl-1b+2014.csv',
            'stats': {
                '2014': { 'name': 'Adrian Gonzalez', 'data': [] },
                '2015': { 'name': 'Paul Goldschmidt', 'data': [] },
                '2016': { 'name': 'Anthony Rizzo', 'data': [] },
                '2017': { 'name': 'Paul Goldschmidt', 'data': [] },
                '2018': { 'name': 'Paul Goldschmidt', 'data': [] },
                '2019': { 'name': 'Freddie Freeman', 'data': [] }
            }
        },
        '2b': {
            'file': 'nl-2b+2014.csv',
            'stats': {
                '2014': { 'name': 'Neil Walker', 'data': [] },
                '2015': { 'name': 'Dee Gordon', 'data': [] },
                '2016': { 'name': 'Daniel Murphy', 'data': [] },
                '2017': { 'name': 'Daniel Murphy', 'data': [] },
                '2018': { 'name': 'Javier Baez', 'data': [] },
                '2019': { 'name': 'Ozzie Albies', 'data': [] }
            }
        },
        '3b': {
            'file': 'nl-3b+2014.csv',
            'stats': {
                '2014': { 'name': 'Anthony Rendon', 'data': [] },
                '2015': { 'name': 'Nolan Arenado', 'data': [] },
                '2016': { 'name': 'Nolan Arenado', 'data': [] },
                '2017': { 'name': 'Nolan Arenado', 'data': [] },
                '2018': { 'name': 'Nolan Arenado', 'data': [] },
                '2019': { 'name': 'Anthony Rendon', 'data': [] }
            }
        },
        'ss': {
            'file': 'nl-ss+2014.csv',
            'stats': {
                '2014': { 'name': 'Ian Desmond', 'data': [] },
                '2015': { 'name': 'Brandom Crawford', 'data': [] },
                '2016': { 'name': 'Corey Seager', 'data': [] },
                '2017': { 'name': 'Corey Seager', 'data': [] },
                '2018': { 'name': 'Trevor Story', 'data': [] },
                '2019': { 'name': 'Trevor Story', 'data': [] }
            }
        },
        'ofs': {
            'file': 'nl-of+2014.csv',
            'stats': {
                '2014': {
                    'names': ['Andrew McCutchen', 'Giancarlo Stanton', 'Justin Upton'],
                    'data': []
                },
                '2015': {
                    'names': ['Andrew McCutchen', 'Bryce Harper', 'Carlos Gonzalez'],
                    'data': []
                },
                '2016': {
                    'names': ['Christian Yelich', 'Yoenis Cespedes', 'Charlie Blackmon'],
                    'data': []
                },
                '2017': {
                    'names': ['Marcell Ozuna', 'Giancarlo Stanton', 'Charlie Blackmon'],
                    'data': []
                },
                '2018': {
                    'names': ['Christian Yelich', 'Nick Markakis', 'David Peralta'],
                    'data': []
                },
                '2019': {
                    'names': ['Cody Bellinger', 'Christian Yelich', 'Ronald Acuna Jr.'],
                    'data': []
                }
            }
        },
        'of1': {
            'file': 'nl-of+2014.csv',
            'stats': {
                '2014': { 'name': 'Andrew McCutchen', 'data': [] },
                '2015': { 'name': 'Andrew McCutchen', 'data': [] },
                '2016': { 'name': 'Christian Yelich', 'data': [] },
                '2017': { 'name': 'Marcell Ozuna', 'data': [] },
                '2018': { 'name': 'Christian Yelich', 'data': [] },
                '2019': { 'name': 'Cody Bellinger', 'data': [] }
            }
        },
        'of2': {
            'file': 'nl-of+2014.csv',
            'stats': {
                '2014': { 'name': 'Giancarlo Stanton', 'data': [] },
                '2015': { 'name': 'Bryce Harper', 'data': [] },
                '2016': { 'name': 'Yoenis Cespedes', 'data': [] },
                '2017': { 'name': 'Giancarlo Stanton', 'data': [] },
                '2018': { 'name': 'Nick Markakis', 'data': [] },
                '2019': { 'name': 'Christian Yelich', 'data': [] }
            }
        },
        'of3': {
            'file': 'nl-of+2014.csv',
            'stats': {
                '2014': { 'name': 'Justin Upton', 'data': [] },
                '2015': { 'name': 'Carlos Gonzalez', 'data': [] },
                '2016': { 'name': 'Charlie Blackmon', 'data': [] },
                '2017': { 'name': 'Charlie Blackmon', 'data': [] },
                '2018': { 'name': 'David Peralta', 'data': [] },
                '2019': { 'name': 'Ronald Acuna Jr.', 'data': [] }
            }
        },
        'c': {
            'file': 'nl-c+2014.csv',
            'stats': {
                '2014': { 'name': 'Buster Posey', 'data': [] },
                '2015': { 'name': 'Buster Posey', 'data': [] },
                '2016': { 'name': 'Wilson Ramos', 'data': [] },
                '2017': { 'name': 'Buster Posey', 'data': [] },
                '2018': { 'name': 'J.T. Realmuto', 'data': [] },
                '2019': { 'name': 'J.T. Realmuto', 'data': [] }
            }
        }
    }
}
predWinners = {}


# In[444]:


def recall(value):
    counter = 0
    for league in winners:
        for position in winners[league]:
            if position == 'of1' or position == 'of2' or position == 'of3' or position == 'of':
                continue
            if position == 'ofs':
                for of in ['of1', 'of2', 'of3']:
                    try:
                        index_elem = predWinners[league][position].index(winners[league][of]['stats']['2019']['name'])
                        if (index_elem+1) <= value * 3:
#                             print(index_elem, winners[league][of]['stats'][str(lastYear + 1)]['name'])
                            counter = counter + 1
                    except ValueError:
                        continue
            else:
                try:
                    index_elem = predWinners[league][position].index(winners[league][position]['stats']['2019']['name'])
                    if (index_elem+1) <= value:
#                         print(index_elem, winners[league][position]['stats']['2019']['name'])
                        counter = counter + 1
                except ValueError:
                    continue
    return counter   


# In[445]:


def ranking():
    accum = 0
    for league in winners:
        for position in winners[league]:
            if position == 'of1' or position == 'of2' or position == 'of3' or position == 'of':
                continue
            if position == 'ofs':
                for of in ['of1', 'of2', 'of3']:
                    try:
                        index_elem = predWinners[league][position].index(winners[league][of]['stats'][str(lastYear + 1)]['name'])
#                         print(index_elem, winners[league][of]['stats']['2019']['name'])
                        accum = accum + 1 / math.ceil(((index_elem + 1) / 3))
                    except ValueError:
                        continue
            else:
                try:
                    index_elem = predWinners[league][position].index(winners[league][position]['stats'][str(lastYear + 1)]['name'])
#                     print(index_elem, winners[league][position]['stats']['2019']['name'])
                    accum = accum + 1 / (index_elem + 1)
                except ValueError:
                    continue
    return accum


# In[446]:


def printStats(mode):
    if mode == 'predicted':
        colList = ['Name', 'runsPredicted', 'HRsPredicted', 'RBIsPredicted', 'AVGPredicted', 'opsPredicted']
    elif mode == 'linearSVR':
        colList = ['Name', 'linearSVRRuns', 'linearSvrHR', 'linearSvrRBI', 'linearSvrAVG', 'linearSvrOPS']
    elif mode == 'svr':
        colList = ['Name', 'svrRuns', 'SvrHR', 'svrRBI', 'svrAVG', 'svrOPS']
    elif mode == 'huber':
        colList = ['Name', 'HuberRegressorRBI', 'HuberRegressorAVG', 'HuberRegressorHR', 'HuberRegressorOPS', 'HuberRegressorRuns']
    elif mode == 'ridge':
        colList = ['Name', 'RidgeRBI', 'RidgeAVG', 'RidgeHR', 'RidgeOPS', 'RidgeRuns']
    else:
        if lastYear == 2018 or lastYear == 2019:
            colList = ['Name', 'runsActual', 'HRsActual', 'RBIsActuals', 'AVGActuals', 'OpsActuals']
        else:
            colList = ['Name', 'runs', 'hr', 'rbi', 'avg', 'ops']
    for league in predWinners:
        for position in predWinners[league]:
            if position == 'ofs':
                for  i in range(0, 3):
                    winner = predWinners[league][position][i]
                    winnerStats = allData.loc[allData['Name'] == winner]
                    winnerStats = winnerStats[colList]
                    print(winnerStats)
            else:
                winner = predWinners[league][position][0]
                winnerStats = allData.loc[allData['Name'] == winner]
                winnerStats = winnerStats[colList]
                print(winnerStats)
            


# In[447]:


# divide players by league and by position
mode = 'ridge'
dataMode = 'scale'
neighbors = 1
for league in specificData:
    for position in specificData[league]:
        position['data'] = getDataByLAP(position['fileName'], mode, dataMode)
compute_by_position(dataMode)
# winners['al']['1b']
print(lastYear+1)
knn(mode, neighbors, False)
# printStats(mode) # uncomment this when you need to see the predicted winner stats
for i in range (1, 6):
    val = recall(i)
    print(val)
print(ranking())


# In[183]:


combinedWinners = pd.concat([winners['al']['1b']['average'], winners['nl']['1b']['average']])
combinedWinners = combinedWinners.mean(axis=0)

