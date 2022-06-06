#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy_financial as npf


# In[2]:


help(npf.pv)


# In[3]:


rate = .09
cashflow= 100000
T= 5


# In[4]:


npf.pv(rate, T,0, -cashflow)


# In[5]:


T =range(1,6)
cashflows= npf.pv(rate, T, 0,  -cashflow)
cashflows


# In[6]:


print(f"{' Year':15}'PV of Cash Flow')")
print(f"-" * 30)
for year in T :
    balance = f'$ {cashflows[year -1]:,.2f}'
    print(f'{year}{balance:>29s}')


# In[7]:


np.cumsum(cashflows).round(2)


# In[ ]:




