#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


get_ipython().system('head UCI_Credit_Card.csv ')


# In[7]:


# (1) Используя параметры pandas прочитать красиво пандас 
df = pd.read_csv('UCI_Credit_Card.csv', sep=',')


# In[4]:


df.head()


# In[8]:


# (2) выведите, что за типы переменных, сколько пропусков,
df.info()


# In[9]:


# для численных значений посчитайте пару статистик (в свободной форме)
df.describe()


# In[12]:


# (3) посчитать число женщин с университетским образованием
# SEX (1 = male; 2 = female). 
# EDUCATION (1 = graduate school; 2 = university; 3 = high school; 4 = others). 


# In[10]:


df[(df['SEX'] == 2)&(df['EDUCATION'] == 2)].shape[0]


# In[14]:


# (4) Сгрупировать по "default.payment.next.month" и посчитать медиану для всех показателей начинающихся на BILL_ и PAY_
#TODO


# In[11]:


bill_pay_list = []
for x in df.columns:
    if x.startswith('BILL') or x.startswith('PAY'):
        bill_pay_list.append(x)


# In[12]:


bill_pay_list_upd = bill_pay_list + ['default.payment.next.month']


# In[13]:


df[bill_pay_list_upd].groupby('default.payment.next.month').median()


# In[14]:


# (5) постройте сводную таблицу (pivot table) по SEX, EDUCATION, MARRIAGE


# In[53]:


d = df.pivot_table(index=['SEX', 'EDUCATION', 'MARRIAGE'])


# In[54]:


d


# In[21]:


# (6) Создать новый строковый столбец в data frame-е, который:
# принимает значение A, если значение LIMIT_BAL <=10000
# принимает значение B, если значение LIMIT_BAL <=100000 и >10000
# принимает значение C, если значение LIMIT_BAL <=200000 и >100000
# принимает значение D, если значение LIMIT_BAL <=400000 и >200000
# принимает значение E, если значение LIMIT_BAL <=700000 и >400000
# принимает значение F, если значение LIMIT_BAL >700000


# In[16]:


def new_col(x):
    if x <= 10000: return 'A'
    elif x >10000 and x <=100000: return 'B'
    elif x >100000 and x <=200000: return 'C'
    elif x >200000 and x <=400000: return 'D'
    elif x >400000 and x <=700000: return 'E'
    else: return 'F'


# In[17]:


df['new_col'] = df['LIMIT_BAL'].apply(new_col)


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


# (7) постироить распределение LIMIT_BAL (гистрограмму)
df['LIMIT_BAL'].hist();


# In[20]:


# (8) построить среднее значение кредитного лимита для каждого вида образования 
# и для каждого пола
# график необходимо сделать очень широким (на весь экран)


# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[29]:


d = df.pivot_table('LIMIT_BAL', 'SEX', 'EDUCATION', 'mean').T.rename({1: 'male', 2: 'female'}, axis=1)
d


# In[35]:


d.plot(figsize=(15,8), legend=True, marker='.', grid=True);


# In[33]:


# (9) построить зависимость кредитного лимита и образования только для одного из полов


# In[34]:


df2['male'].plot(figsize=(15,8), legend=True, marker='.', grid=True);


# In[36]:


# (10) построить большой график (подсказка - используя seaborn) для построения завимисости всех возможных пар параметров
# разным цветом выделить разные значение "default payment next month"
# (но так как столбцов много - картинка может получиться "монструозной")
# (поэкспериментируйте над тем как построить подобное сравнение параметров)
# (подсказка - ответ может состоять из несколькольких графиков)
# (если не выйдет - программа минимум - построить один график со всеми параметрами)


# In[37]:


df.head()


# ### v1

# In[38]:


hue_col = 'default.payment.next.month'


# In[52]:


sns.pairplot(df.drop(['ID'], axis=1), hue=hue_col)


# In[ ]:




