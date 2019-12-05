
# coding: utf-8

# In[32]:


import pandas as pd
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[33]:


df = pd.read_excel('2019PredictionsAndActual.xlsx', index_col=0)
ndf = df.dropna()


# In[34]:


ndf = ndf.rename(columns={'RBIsActuals': 'RBIsActual', 'AVGActuals': 'AVGActual', 'OpsActuals': 'OpsActual'})


# In[35]:


cols = ['runs', 'HRs', 'RBIs', 'AVG', 'Ops']
mae = {}
rmse = {}


# In[36]:


for col in cols:
    print(col)
    mae[col] = mean_absolute_error(ndf[col + 'Actual'], ndf[col + 'Predicted'])
    rmse[col] = math.sqrt(mean_squared_error(ndf[col + 'Actual'], ndf[col + 'Predicted']))
 
    


# In[39]:


print(mae)


# In[40]:


print(rmse)


# In[42]:


dd = pd.read_csv('2019actual.csv')


# In[43]:


dd

