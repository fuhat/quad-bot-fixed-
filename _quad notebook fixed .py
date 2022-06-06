#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np 
from matplotlib import rcParams
rcParams['figure.figsize']=8, 6
import seaborn as sb 
sb.set


# In[14]:


import matplotlib.pyplot as plt


# In[15]:


from matplotlib import rcParams 
rcParams['figure.figsize']= 8,6
import seaborn as sb
sb.set()


# In[16]:


import pandas_datareader as pdr


# In[17]:


amzn = pdr.get_data_yahoo('AMZN')


# In[18]:


import pandas as pd


# In[19]:


amzn = pdr.get_data_yahoo('AMZN')


# In[20]:


import yfinance as yf


# In[21]:


import pandas_datareader as pdr 


# In[22]:


amzn.head()


# In[23]:


amzn_close= amzn['Close']
amzn_return = round (np.log(amzn_close).diff() *100, 2)
amzn_return.head()


# In[24]:


import numpy as np


# In[25]:


amzn_return[-252:].plot()


# In[26]:


amzn_return.dropna(inplace =True)
amzn_return.describe()


# In[27]:


from scipy import stats 


# In[28]:


n, minimax, mean, var, skew, kurt = stats.describe(amzn_return)
mini, maxi= minimax
std= var ** .5


# In[29]:


plt.hist(amzn_return, bins = 15);


# In[30]:


from scipy.stats import norm
x= norm.rvs(mean, std,n)


# In[31]:


plt.hist(x, bins =15);


# In[32]:


x_test = stats.kurtosistest(x)
amzn_test = stats.kurtosistest(amzn_return)
print(f'{" test statistic":20}{"p-value":>15}')
print(f' {" "*5}{"-"*30}')
print(f"x:{x_test[0]:>17.2f}{x_test[1]:16.4f}")
print(f"AMZN: {amzn_test[0]:13.2f}{amzn_test[1]:16.4f}")


# In[33]:


plt.hist(amzn_return, bins= 25 , edgecolor= 'w', density = True)
overlay = np.linspace(mini, maxi, 100)
plt.plot(overlay, norm.pdf(overlay,mean, std));


# In[34]:


stats.ttest_1samp(amzn_return.sample(252), 0, alternative = 'two-sided')


# In[39]:


amzn_close = pd.DataFrame(amzn_close , columns = ['Close'])
amzn_close['lag_1'] = amzn_close.Close.shift(1)
amzn_close['lag_2'] = amzn_close.Close.shift(2)
amzn_close.dropna(inplace = True)
amzn_close.tail()


# In[40]:


lr = np.linalg.lstsq(amzn_close[['lag_1', 'lag_2']], amzn_close['Close'], rcond = None)[0]


# In[41]:


amzn_close['predict'] = np.dot(amzn_close[[ 'lag_1', 'lag_2']], lr)


# In[42]:


amzn_close.tail()


# In[43]:


amzn_close[['Close','predict']].plot()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




