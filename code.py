#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn  as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#read Cancer Data Useing Pandas Library
data=pd.read_csv("cancer data.csv")
data.head()


# In[3]:


data.median()


# In[4]:


data.mean()


# In[5]:


#Data Information
print(data.keys())
data.info()


# In[6]:


data.plot()


# In[7]:


#countplot of Output Variable
sns.countplot(data['benign_0__mal_1'])


# In[8]:


#cheacking Nan or Null data in dataset
data.isnull().sum()


# In[9]:


#Nan data or Null Data Visualization Using Heatmap
sns.heatmap(data.isnull())


# In[10]:


#clean Data 
data1=data.dropna(axis=0)
data1.isnull().sum()


# In[11]:


data1.to_csv("clean.csv")


# In[10]:


#clean data Visualization
sns.heatmap(data1.isnull())
data.shape


# ### Feature Selection

# In[15]:


#Inpute Variables 12
x=data1.iloc[:,0:13].values

#output Variable Benign Or Mal
y=data1.iloc[:,13:].values

#shape of the input and Output Varibale
print("shape of the input Var",x.shape)


print("shape of the output Va",y.shape)


# ### DATA Spliting

# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
x_test[5]


# ### Model Training Using DT,RF,KNN,SVM

# In[30]:


#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
Dt=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=None, min_samples_split=2,min_samples_leaf=1,)
Dt.fit(x_train,y_train)
dt_score=Dt.score(x_test,y_test)*100
print("Accuracy of the Model Is ------>>",dt_score,"%")
pread=Dt.predict(x_test)

fram={'actual_value':[y_test[9]],
      'predicted_value':[pread[9]]}
all1=pd.DataFrame(fram)
all1.head()


# In[31]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier( n_estimators=100,criterion='gini',max_depth=None, min_samples_split=2,min_samples_leaf=1 )
rf.fit(x_train,y_train)
rf_score=rf.score(x_train,y_train)*100
print("Accuracy Of Random forest------->>>",rf_score,"%")
rf.predict(x_test)


# In[34]:


#KNN 
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,weights='uniform',algorithm='auto',leaf_size=3)
knn.fit(x_train,y_train)
knn_score=knn.score(x_train,y_train)*100
print("Accuracy of the KNN",knn_score,"k")
knn.predict(x_test)


# In[53]:


#Svm
from sklearn.svm import SVC
svc=SVC(C=1.0,kernel='rbf')
svc.fit(x_train,y_train)
svc_score=svc.score(x_train,y_train)
print("Accuracy Of SVM Is------>>>",svc_score*100,"%")
svc.predict(x_test)


# In[49]:


df={"Model_Name":["Decision Tree","Random Forest","KNN","SVM"],
    "Accuracy":[dt_score,rf_score,knn_score,svc_score]}
pd.DataFrame(df)


# ### Best Accuracy Model Saved (RF Model)

# In[51]:


import joblib
joblib.dump(rf,"model.pkl")


# In[52]:


model=joblib.load("model.pkl")
model.predict(x_test)

