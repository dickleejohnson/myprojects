#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Big Data Comp Systems (COSC 5340-47F)

# #                           Richard Johnson L20455045
# 
# 
# "Auto MPG Dataset"

# This dataset is a slightly modified version of the dataset provided in the StatLib library. In line with the use by Ross Quinlan (1993) in predicting the attribute "mpg", 8 of the original instances were removed because they had unknown values for the "mpg" attribute. The original dataset is available in the file "auto-mpg.data-original".
# 
# "The data concerns city-cycle fuel consumption in miles per gallon, to be predicted in terms of 3 multivalued discrete and 5 continuous attributes." (Quinlan, 1993)
# 

#  Attribute Information:
# 1. mpg: continuous
# 2. cylinders: multi-valued discrete
# 3. displacement: continuous
# 4. horsepower: continuous
# 5. weight: continuous
# 6. acceleration: continuous
# 7. model year: multi-valued discrete
# 8. origin: multi-valued discrete
# 9. car name: string (unique for each instance)

# In[1]:


#Importing required library
import numpy as np
import pandas as pd


# In[14]:


#Load 'car.data' dataset into a pandas dataframe object 'data'

data = pd.read_csv('auto-mpg.data-original', delim_whitespace=True, header=None)


# In[15]:


#Checking top5 data 
data.head()


# In[28]:


#Setting the columns name from our dataset file

data.columns = ['mpg','cylinders','displacement','hp','weight','acceleration', 'model_year', 'origin', 'car_name']


# In[29]:


#Watching the changes
data


# In[30]:


#Getting information of our dataset
data.info()


# In[31]:


#Checking for null values 
#Result shows no null values
data.isnull().any()


# In[32]:


#Finding number of unique value in 'mpg' column

data['mpg'].nunique()


# In[33]:


#Finding unique values for all columns

data['cylinders'].unique()


# In[34]:


data['displacement'].unique()


# In[35]:


data['hp'].unique()


# In[36]:


data['weight'].unique()


# In[37]:


data['acceleration'].unique()


# In[38]:


data['model_year'].unique()


# In[39]:


data['origin'].unique()


# In[40]:


data['car_name'].unique()


# In[41]:


#Finding unique value in each column
data.nunique()


# # Handling Missing Values

# In[42]:


#Defining method to replace '?' with null value

def check(x):
    if x == '?':
        return np.NaN
    return x


# In[43]:


#Applying that function to every columns

data['mpg'] = data['mpg'].apply(check)
data['cylinders'] = data['cylinders'].apply(check)
data['displacement'] = data['displacement'].apply(check)
data['hp'] = data['hp'].apply(check)
data['weight'] = data['weight'].apply(check)
data['acceleration'] = data['acceleration'].apply(check)
data['model_year'] = data['model_year'].apply(check)
data['origin'] = data['origin'].apply(check)
data['car_name'] = data['car_name'].apply(check)


# In[44]:


#Checking total null value in each columns

data.isnull().sum()


# In[45]:


#Finding average value of 'mpg' after dropping null value and converting rest of the objects to integer
avg_col1 = round(data['mpg'].dropna().astype(int).mean())


# In[46]:


avg_col1


# In[47]:


#Handling missing values

#Finding average value of each columns and replacing null value with average value of each columns
avg_col = []
for i in range(len(data.columns)-1):
    avg_col.append(round(data[data.columns[i]].dropna().astype(int).mean()))
    data[data.columns[i]] = data[data.columns[i]].fillna(avg_col[i])
    print(avg_col[i])


# In[48]:


#Converting all the attribute columns to integer type

data['mpg'] = data['mpg'].astype(int)
data['cylinders'] = data['cylinders'].astype(int)
data['displacement'] = data['displacement'].astype(int)
data['hp'] = data['hp'].astype(int)
data['weight'] = data['weight'].astype(int)
data['acceleration'] = data['acceleration'].astype(int)
data['model_year'] = data['model_year'].astype(int)
data['origin'] = data['origin'].astype(int)


# In[49]:


#Checking the data again

data.info()


# In[50]:


# Check for any number of missing value
data.isnull().sum()


# # Calculate Similarity for each attribute

# ![](1.png)

# In[61]:


#Creating numpy array equals the size of data for each columns
#To store the similarity between data objects

array1 = np.zeros((len(data),len(data)))
array2 = np.zeros((len(data),len(data)))
array3 = np.zeros((len(data),len(data)))
array4 = np.zeros((len(data),len(data)))
array5 = np.zeros((len(data),len(data)))
array6 = np.zeros((len(data),len(data)))
array7 = np.zeros((len(data),len(data)))
array8 = np.zeros((len(data),len(data)))
array9 = np.zeros((len(data),len(data)))


# In[62]:


array1


# In[63]:


#Similarity for the attribute 'mpg' (RATIO ATTRIBUTE)

#Using d = |p-q|; s = 1/(1+d)

s = data['mpg']

for i in range(0,406):
    for j in range (i,406):
        array1[i,j]  =  1/(1+(abs (s[i] - s[j])))
        array1[j,i] = 1/(1+(abs (s[i] - s[j])))

print(array1)


# In[64]:


#Similarity for the attribute 'cylinders' (RATIO ATTRIBUTE)

#Using d = |p-q|; s = 1/(1+d)

s = data['cylinders']

for i in range(0,406):
    for j in range (i,406):
        array2[i,j]  =  1/(1+(abs (s[i] - s[j])))
        array2[j,i] = 1/(1+(abs (s[i] - s[j])))

print(array2)


# In[65]:


#Similarity for the attribute 'displacement' (RATIO ATTRIBUTE)

#Using d = |p-q|; s = 1/(1+d)

s = data['displacement']

for i in range(0,406):
    for j in range (i,406):
        array3[i,j]  =  1/(1+(abs (s[i] - s[j])))
        array3[j,i] = 1/(1+(abs (s[i] - s[j])))

print(array3)


# In[66]:


#Similarity for the attribute 'hp' (RATIO ATTRIBUTE)

#Using d = |p-q|; s = 1/(1+d)

s = data['hp']

for i in range(0,406):
    for j in range (i,406):
        array4[i,j]  =  1/(1+(abs (s[i] - s[j])))
        array4[j,i] = 1/(1+(abs (s[i] - s[j])))

print(array4)


# In[67]:


#Similarity for the attribute 'weight' (RATIO ATTRIBUTE)

#Using d = |p-q|; s = 1/(1+d)

s = data['weight']

for i in range(0,406):
    for j in range (i,406):
        array5[i,j]  =  1/(1+(abs (s[i] - s[j])))
        array5[j,i] = 1/(1+(abs (s[i] - s[j])))

print(array5)


# In[68]:


#Similarity for the attribute 'acceleration' (RATIO ATTRIBUTE)

#Using d = |p-q|; s = 1/(1+d)

s = data['acceleration']

for i in range(0,406):
    for j in range (i,406):
        array6[i,j]  =  1/(1+(abs (s[i] - s[j])))
        array6[j,i] = 1/(1+(abs (s[i] - s[j])))

print(array6)


# In[69]:


#Similarity for the attribute 'model_year' (RATIO ATTRIBUTE)

#Using d = |p-q|; s = 1/(1+d)

s = data['model_year']

for i in range(0,406):
    for j in range (i,406):
        array7[i,j]  =  1/(1+(abs (s[i] - s[j])))
        array7[j,i] = 1/(1+(abs (s[i] - s[j])))

print(array7)


# In[70]:


#Similarity for the attribute 'origin' (RATIO ATTRIBUTE)

#Using d = |p-q|; s = 1/(1+d)

s = data['origin']

for i in range(0,406):
    for j in range (i,406):
        array8[i,j]  =  1/(1+(abs (s[i] - s[j])))
        array8[j,i] = 1/(1+(abs (s[i] - s[j])))

print(array8)


# In[73]:


#Similarity for the attribute  'car_name'(NOMINAL ATTRIBUTE)

#Using s = 1 if p=q; s=0 if p!=q

s = data['car_name']

for i in range(0,406):
    for j in range (i,406):
        if s [i] == s[j]:
            array9[i,j] = 1
            array9[j,i] = 1
    
        else:
            array9[i,j] = 0
            array9[j,i] = 0

print(array9)


# # Combining Similarity

# ![](2.png)

# In[74]:


#Creating Dataframe to store similairty value of each data objects attribute-wise

mat1 = pd.DataFrame(array1)
mat2 = pd.DataFrame(array2)
mat3 = pd.DataFrame(array3)
mat4 = pd.DataFrame(array4)
mat5 = pd.DataFrame(array5)
mat6 = pd.DataFrame(array6)
mat7 = pd.DataFrame(array7)
mat8 = pd.DataFrame(array8)
mat9 = pd.DataFrame(array9)
z = np.zeros((len(data),len(data)))
final_matrix = pd.DataFrame(z)


# In[75]:


# Using combining similiraty formula to combine the similarity between attributes
# In formula, del(k) value equals 1 because the values does not equals zero or have missing values

d = 1
for i in range(len(data)):
    for j in range(i,len(data)):
        z[i,j] = z[j,i] = (d*mat1.loc[i,j]+d*mat2.loc[i,j]+d*mat3.loc[i,j]+d*mat4.loc[i,j]+d*mat5.loc[i,j]
                          +d*mat6.loc[i,j]+d*mat7.loc[i,j]+d*mat8.loc[i,j]+d*mat9.loc[i,j])/9


# In[76]:


#Checking combined similarity value
z


# In[77]:


#Creating Dataframe for the final values
final_matrix = pd.DataFrame(z)


# In[78]:


final_matrix


# In[79]:


#Writing the combined similarity to output file
final_matrix.to_csv('Similarity Matrix.csv')

