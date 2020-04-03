#!/usr/bin/env python
# coding: utf-8

# In[1]:


prices_a = [8.70, 8.91, 8.71]


# In[2]:


8.91/8.70 -1 


# In[3]:


prices_a [1:]


# In[4]:


prices_a[:-1]


# In[14]:


#Methode array 
import numpy as np 


# In[16]:


prices_a = np.array([8.70, 8.91, 8.71])
prices_a 


# In[18]:


prices_a[1:]/prices_a[:-1]-1


# In[6]:


#Methode panda
import pandas as pd
prices=pd.DataFrame({"Stock A":[8.70, 8.91, 8.71, 8.43, 8.73],
                    "Stock B":[10.66, 11.08, 10.71, 11.59, 12.11]
                    })


# In[9]:


prices


# In[10]:


prices.iloc[1:]


# In[11]:


prices.iloc[:-1]


# In[12]:


prices.iloc[1:]/prices.iloc[:-1]  #not correct because of alignment 


# In[13]:


prices.iloc[1:].values/prices.iloc[:-1]


# In[33]:


prices.iloc[1:]/prices.iloc[:-1].values


# In[34]:


prices


# In[36]:


prices.shift(1)


# In[38]:


prices/prices.shift(1) -1 


# In[39]:


prices 


# In[40]:


prices.pct_change()


# In[14]:


returns = prices.pct_change()


# In[15]:


returns 


# In[18]:


prices.plot(title='Returns of Stock A and Stock B')


# In[48]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[52]:


returns.plot.bar()


# In[53]:


returns.head()


# In[54]:


returns.std()


# In[55]:


returns.mean()


# In[56]:


returns


# In[20]:


import numpy as np
np.prod(returns+1)


# In[21]:


#


# In[ ]:




