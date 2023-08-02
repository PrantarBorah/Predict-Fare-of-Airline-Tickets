#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train_data=pd.read_excel(r"C:\Users\Manjit.Borah\Downloads/Data_Train.xlsx")


# In[3]:


train_data.info()


# In[4]:


train_data.isnull().sum()


# In[5]:


train_data.dropna(inplace=True)


# In[6]:



train_data.isnull().sum()


# In[7]:


train= train_data.copy()


# In[8]:


train.dtypes


# In[9]:


def change_into_datetime(col):
    train[col]=pd.to_datetime(train[col])
    


# In[10]:


train.columns


# In[11]:


for feature in ['Date_of_Journey','Dep_Time','Arrival_Time']:
    change_into_datetime(feature)


# In[12]:


train.dtypes


# In[13]:


train['Date_of_Journey'].min()


# In[14]:


train['journey_day']= train['Date_of_Journey'].dt.day


# In[15]:


train['journey_month']= train['Date_of_Journey'].dt.month


# In[16]:


train['journey_year']= train['Date_of_Journey'].dt.year


# In[17]:


train.head(3)


# In[18]:


train.drop('Date_of_Journey', axis=1, inplace=True)


# In[19]:


def extract_hour_min(df,col):
    df[col+'_hour']=df[col].dt.hour
    df[col+'_min']=df[col].dt.minute
    df.drop(col,axis=1,inplace=True)
    return df.head(2)


# In[20]:


extract_hour_min(train,'Dep_Time')


# In[21]:


extract_hour_min(train,'Arrival_Time')


# In[22]:


#When will the flights take off (DATA_ANALYSIS)


# In[23]:


def flight_dep_time(x):
    
        if(x>4) and (x<=8):
            return 'early morning'
        elif(x>8) and (x<=12):
            return 'morning'
        elif(x>12) and (x<=16):
            return 'noon'
        elif(x>16) and (x<=20):
            return 'evening'
        elif(x>20) and (x<=24):
            return 'night'
        else:
            return 'late night'
        


# In[24]:


train['Dep_Time_hour'].apply(flight_dep_time).value_counts().plot(kind='bar')


# In[25]:


pip install chart_studio


# In[26]:


import plotly
import cufflinks as cf
from cufflinks.offline import go_offline


# In[27]:


def preprocess_duration(x):
    if 'h' not in x:
        x='0h '+x
    elif 'm' not in x:
        x=x+' 0m'
    return x


# In[28]:


train['Duration']=train['Duration'].apply(preprocess_duration)


# In[29]:


train['Duration_hour']=train['Duration'].apply(lambda x:int(x.split(' ')[0][0:-1]))


# In[30]:


int(train['Duration'][0].split(' ')[1][0:-1])


# In[31]:


train['Duration_mins']=train['Duration'].apply(lambda x:int(x.split(' ')[1][0:-1]))


# In[32]:


train.head()


# In[33]:


train['Duration_in_mins']=train['Duration'].str.replace('h','*60').str.replace(' ','+').str.replace('m','*1').apply(eval)


# In[34]:


train.head(2)


# In[35]:


sns.lmplot(x="Duration_in_mins",y='Price', data=train)


# In[36]:


train['Destination'].value_counts().plot(kind='pie')


# In[37]:


train['Route']


# In[38]:


plt.figure(figsize=(15,5))
sns.boxplot(y='Price', x='Airline',data=train)
plt.xticks(rotation='vertical')


# In[39]:


plt.figure(figsize=(15,5))
sns.violinplot(y='Price', x='Airline',data=train)
plt.xticks(rotation='vertical')


# In[40]:


train['Additional_Info'].value_counts()


# In[41]:


train.drop(columns=['Additional_Info','Route','Duration_in_mins','journey_year'],axis=1,inplace=True)


# In[42]:


train.info()


# In[43]:


categorical_column=[col for col in train.columns if train[col].dtype=='object']


# In[44]:


numerical_column=[col for col in train.columns if train[col].dtype!='object']


# In[45]:


categorical_column


# In[46]:


numerical_column


# In[47]:


for category in train['Source'].unique():
    train['Source_'+category]= train['Source'].apply(lambda x: 1 if x==category else 0)
    


# In[48]:


airlines=train.groupby(['Airline'])['Price'].mean().sort_values().index


# In[49]:


airlines


# In[50]:


dict1={key:index for index,key in enumerate(airlines,0)}


# In[51]:


dict1


# In[52]:


train['Airline']=train['Airline'].map(dict1)


# In[53]:


train.head(2)


# In[54]:


train['Destination'].unique()


# In[55]:


train['Destination'].replace('New Delhi','Delhi',inplace=True)


# In[56]:


dest=train.groupby(['Destination'])['Price'].mean().sort_values().index


# In[57]:


dict2={key:index for index,key in enumerate(dest,0)}


# In[58]:


train['Destination']= train['Destination'].map(dict2)


# In[59]:


train.head()


# In[60]:


stops={'non-stop':0,'2 stops':2, '1 stop':1,'3 stops':3,'4 stops':4}


# In[61]:


train['Total_Stops']= train['Total_Stops'].map(stops)


# In[62]:


train.head()


# In[63]:


def plot(df,col) :
   fig,(ax1,ax2,ax3)=plt.subplots(3,1)
   sns.distplot(df[col],ax=ax1)
   sns.boxplot(df[col],ax=ax2)
   sns.distplot(df[col],ax=ax3,kde=False)


# In[64]:


plot(train,'Price')


# In[65]:


train['Price']=np.where(train['Price']>=35000, train['Price'].median(),train['Price'])


# In[66]:


plot(train,'Price')


# In[67]:


train.drop(columns=['Source','Duration'],axis=1,inplace=True)


# In[68]:


train.dtypes


# In[69]:


from sklearn.feature_selection import mutual_info_regression


# In[70]:


X= train.drop(['Price'],axis=1)


# In[71]:


Y= train['Price']


# In[72]:


mutual_info_regression(X,Y)


# In[73]:


imp=pd.DataFrame(mutual_info_regression(X,Y),index=X.columns)


# In[74]:


imp


# In[75]:


#HIgh variance to Low variance ( Random Forrest)


# In[76]:


from sklearn.model_selection import train_test_split


# In[77]:


X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.3, random_state=42)


# In[78]:


from sklearn.ensemble import RandomForestRegressor


# In[79]:


ml_model= RandomForestRegressor()


# In[80]:


model= ml_model.fit(X_train,Y_train)


# In[81]:


y_pred=model.predict(X_test)


# In[82]:


y_pred


# In[83]:


get_ipython().system('pip install pickle')


# In[84]:


import pickle


# In[85]:


file=open(r'C:\Users\Manjit.Borah\Downloads/rf_ranmod.pkl','wb')


# In[86]:


pickle.dump(model,file)


# In[87]:


def mape(Y_true,Y_pred):
    Y_true,Y_pred= np.array(Y_true),np.array(Y_pred)
    
    return np.mean(np.abs((Y_true-Y_pred)/Y_true))*100


# In[88]:


model= open(r'C:\Users\Manjit.Borah\Downloads/rf_ranmod.pkl','rb') 


# In[89]:


forest= pickle.load(model)


# In[90]:


forest.predict(X_test)


# In[91]:


mape(Y_test,forest.predict(X_test))


# In[103]:


#def predict(ml_model):
    #model=ml_model.fit(X_train,Y,train)
    #print('Training_score')
from sklearn import metrics


# In[106]:


def predict(ml_model):
    model = ml_model.fit(X_train, Y_train)
    print('Training Score: {}'.format(model.score(X_train, Y_train)))
    pred_df = pd.DataFrame({'Actual' : Y_test, 'Predicted' : y_pred})
    print('\n')
    print(pred_df)
    print('\n')
    print('r2_score: {}'.format(metrics.r2_score(Y_test, y_pred)))
    print('MAE: {}'.format(metrics.mean_absolute_error(Y_test, y_pred)))
    print('MSE: {}'.format(metrics.mean_squared_error(Y_test, y_pred)))
    print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(Y_test, y_pred))))
    
    sns.displot(Y_test - y_pred)


# In[107]:


predict(RandomForestRegressor())


# In[108]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# In[109]:


predict(LinearRegression())


# In[110]:


predict(KNeighborsRegressor())


# In[111]:


predict(DecisionTreeRegressor())


# In[ ]:




