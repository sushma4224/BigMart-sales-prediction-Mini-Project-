#!/usr/bin/env python
# coding: utf-8

# # BigMart Sales (Predicting sales of the product)

# # Problem Statement :
# The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also,certain attributes of each product and store have been defined. The aim is to build a predictive model and find out the saless of each product at a particular store.

# Importing all necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings ('ignore')


# Loading the Data set:
# We have train(8523) and test(5681) data set,train data set has both input and output variables. we need to predict the sales for test data set.
# Combine test and train into one file.

# In[2]:


Train= pd.read_csv("C:/Users/murthy/OneDrive/Desktop/Train.csv")
Test = pd.read_csv("C:/Users/murthy/OneDrive/Desktop/Test.csv")
Train['source']='Train'
Test['source']='Test'
df=pd.concat([Train,Test],ignore_index=True)


# In[3]:


df


# In[4]:


df.head()


# Statistical Information

# In[5]:


df.describe()


#  Checking Data type of Attributes

# In[6]:


df.info()


# We have Categorical as well as numerical attributes which we will process seperately.

# Checking Unique values in dataset :

# In[7]:


df.apply(lambda x: len(x.unique()))


# Attributes containing many unique values are of numerical type. The remaining attributes are of categorical type.

# Preprocessing the Dataset

# Checking Missing (null) values:

# In[8]:


df.isnull().sum()


# We observe three attributes with many missing values (Item_Outlet_Sales, Item_Weight and Outlet_Size)

# Fill Missing values using Mean and Mode funcion:

# In[9]:


df['Item_Outlet_Sales'] = df['Item_Outlet_Sales'].fillna(df['Item_Outlet_Sales'].mode()[0])
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])


# In[10]:


df


# All the missing values are now filled.

# Checking again wether there is still have missing values or not

# In[11]:


df.isnull().sum()


# In[12]:


df.info()


# Data Cleaning:

# Removing unnecessary columns from the dataset

# In[13]:


df=df.drop(['Item_Identifier', "Outlet_Identifier"],axis=1)
df.head()


# Exploratatory Data Analysis(EDA):

# Explore the Numerical columns

# In[14]:


sns.distplot(df["Item_Weight"])


# we Observed higher mean values

# In[15]:


sns.distplot(df["Item_Visibility"])


# All the values are small and it shows a left-skewed curve.

# In[16]:


sns.distplot(df["Item_MRP"])


# This graph shows four peak values.

# In[17]:


sns.distplot(df["Item_Outlet_Sales"])


# The values are high and the curve is left_skewed.

# Explore the Categorical columns.

# In[18]:


sns.countplot(df["Item_Fat_Content"])


# In[19]:


sns.countplot(df['Outlet_Establishment_Year'])


# Most outlets are established in an equal distribution

# Label Encoding:

# Label Encoding is to convert the categorical column into the numerical column.

# In[20]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in df.columns:
    df[col] = labelencoder.fit_transform(df[col])


# One Hot Encoding:

# we can also use one hot encoding for categorical columns.

# In[21]:


df=pd.get_dummies(df,columns=['Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type'],drop_first=False)


# In[22]:


df


# It will create a new column for each category. Hence, it will add the corresponding category instead of numerical values.
# if the corresponding location type i present it will show as "1" orelse it will show "0"

# In[23]:


# It will show  all the column names in the dataset
df.columns


# Splitting the data for Training and Testing.

# Before training the model let us drop some unnecessary columns

# In[24]:


X= df.drop(columns=["Outlet_Establishment_Year","Item_Outlet_Sales"])
y=df["Item_Outlet_Sales"]


# X contains input attributes and y contains the output attribute.

# In[25]:


X


# In[26]:


y


# Model Training :

# Now the preprocessing has been done, perform the model training and testing. 

# In[27]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)


# In[28]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import cross_val_Predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate


# Linear Regression

# In[29]:


lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)


# Predicting the values using Linear regression model

# In[30]:


print(X_test) # Testing data 
y_pred =lin_reg.predict(X_test) 


# In[31]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[32]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[33]:


from sklearn.metrics import r2_score
r2= r2_score(y_test,y_pred)
print("R2 score:", r2)


# Random Forest

# In[34]:


RForest=RandomForestRegressor(n_jobs=-1)
RForest.fit(X_train,y_train)


# Predicting the values using Random Forest model.

# In[35]:


#print(X_test) # Testing data 
y_pred =RForest.predict(X_test) 
print(y_pred)


# In[36]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# In[37]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[38]:


from sklearn.metrics import r2_score
r2= r2_score(y_test,y_pred)
print("R2 score:", r2)


# Summary:

# The model achieved an R-squared(R2) value of 0.65 suggests that 65% of the variance in sales can be explained by the features used in the model.
# The model's accuracy is reasonably good, but there is room for improvement. Fine-tuning hyperparameters and considering other algorithms may enhance the model's performance.

# In[39]:


#pip install xgboost


# In[40]:


#from xgboost import XGBRegressor
#xgb=XGBRegressor(learning_rate=0.05,n_estimators=1000)
#xgb.fit(X_train,y_train)

