
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


COLUMN_NAMES = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 
                'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'Class']

DATA = pd.read_csv('crx.csv', names=COLUMN_NAMES)


# In[3]:


DATA.head()


# In[4]:


DATA.info()


# In[5]:


DATA.describe()


# In[6]:


COLUMN_WITH_MISSING = ['A1', 'A2', 'A4', 'A5', 'A6', 'A7', 'A14']

for c in COLUMN_WITH_MISSING:
    print(DATA[c].value_counts())


# In[7]:


# #from the above we can see that the '?' indicates missing values
# #We convert it into NaN

for c in COLUMN_WITH_MISSING:  
    DATA[c] = DATA[c].replace('?', np.NAN)
#    print(data[c].value_counts())


# In[8]:


#we have represented our missing values clearly
DATA.info()


# In[9]:


#replace missing values

NUM_COL = ['A2', 'A14']
for c in NUM_COL:
    DATA[c] = DATA[c].astype('float64')
    MEAN = DATA[c].mean()
    DATA[c] = DATA[c].replace(np.NaN, mean)


DATA['A1'] = DATA['A1'].replace(np.NaN, 'b')
DATA['A4'] = DATA['A4'].replace(np.NaN, 'u')
DATA['A5'] = DATA['A5'].replace(np.NaN, 'g')
DATA['A6'] = DATA['A6'].replace(np.NaN, 'c')
DATA['A7'] = DATA['A7'].replace(np.NaN, 'v')


# In[ ]:


#handling class column

DATA['Class'] = DATA['Class'].replace("+", 1)
DATA['Class'] = DATA['Class'].replace('-', 0)


# In[ ]:


DATA.head()


# In[ ]:


DATA.info()


# In[ ]:


DATA.to_csv('cleaned_data.csv')

