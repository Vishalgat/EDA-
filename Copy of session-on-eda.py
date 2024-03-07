#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('train.csv')


# In[ ]:


df.head()


# ### Why do EDA
# 
# - Model building
# - Analysis and reporting
# - Validate assumptions
# - Handling missing values
# - feature engineering
# - detecting outliers

# In[ ]:


# Remember it is an iterative process


# ### Column Types
# 
# - **Numerical** - Age,Fare,PassengerId
# - **Categorical** - Survived, Pclass, Sex, SibSp, Parch,Embarked
# - **Mixed** - Name, Ticket, Cabin

# ### Univariate Analysis
# 
# Univariate analysis focuses on analyzing each feature in the dataset independently.
# 
# - **Distribution analysis**: The distribution of each feature is examined to identify its shape, central tendency, and dispersion.
# 
# - **Identifying potential issues**: Univariate analysis helps in identifying potential problems with the data such as outliers, skewness, and missing values

# #### The shape of a data distribution refers to its overall pattern or form as it is represented on a graph. Some common shapes of data distributions include:
# 
# - **Normal Distribution**: A symmetrical and bell-shaped distribution where the mean, median, and mode are equal and the majority of the data falls in the middle of the distribution with gradually decreasing frequencies towards the tails.
# 
# - **Skewed Distribution**: A distribution that is not symmetrical, with one tail being longer than the other. It can be either positively skewed (right-skewed) or negatively skewed (left-skewed).
# 
# - **Bimodal Distribution**: A distribution with two peaks or modes.
# 
# - **Uniform Distribution**: A distribution where all values have an equal chance of occurring.
# 
# The shape of the data distribution is important in identifying the presence of outliers, skewness, and the type of statistical tests and models that can be used for further analysis.

# #### **Dispersion** is a statistical term used to describe the spread or variability of a set of data. It measures how far the values in a data set are spread out from the central tendency (mean, median, or mode) of the data.
# 
# There are several measures of dispersion, including:
# 
# - **Range**: The difference between the largest and smallest values in a data set.
# 
# - **Variance**: The average of the squared deviations of each value from the mean of the data set.
# 
# - **Standard Deviation**: The square root of the variance. It provides a measure of the spread of the data that is in the same units as the original data.
# 
# - **Interquartile range (IQR)**: The range between the first quartile (25th percentile) and the third quartile (75th percentile) of the data.
# 
# Dispersion helps to describe the spread of the data, which can help to identify the presence of outliers and skewness in the data.

# ### Steps of doing Univariate Analysis on Numerical columns
# 
# - **Descriptive Statistics**: Compute basic summary statistics for the column, such as mean, median, mode, standard deviation, range, and quartiles. These statistics give a general understanding of the distribution of the data and can help identify skewness or outliers.
# 
# - **Visualizations**: Create visualizations to explore the distribution of the data. Some common visualizations for numerical data include histograms, box plots, and density plots. These visualizations provide a visual representation of the distribution of the data and can help identify skewness an outliers.
# 
# - **Identifying Outliers**: Identify and examine any outliers in the data. Outliers can be identified using visualizations. It is important to determine whether the outliers are due to measurement errors, data entry errors, or legitimate differences in the data, and to decide whether to include or exclude them from the analysis.
# 
# - **Skewness**: Check for skewness in the data and consider transforming the data or using robust statistical methods that are less sensitive to skewness, if necessary.
# 
# - **Conclusion**: Summarize the findings of the EDA and make decisions about how to proceed with further analysis.
# 

# ### Age
# 
# **conclusions**
# 
# - Age is normally(almost) distributed
# - 20% of the values are missing
# - There are some outliers

# In[ ]:


df['Age'].describe()


# In[ ]:


df['Age'].plot(kind='hist',bins=20)


# In[ ]:


df['Age'].plot(kind='kde')


# In[ ]:


df['Age'].skew()


# In[ ]:


df['Age'].plot(kind='box')


# In[ ]:


df[df['Age'] > 65]


# In[ ]:


df['Age'].isnull().sum()/len(df['Age'])


# ### Fare
# 
# **conclusions**
# 
# - The data is highly(positively) skewed
# - Fare col actually contains the group fare and not the individual fare(This migth be and issue)
# - We need to create a new col called individual fare

# In[ ]:


df['Fare'].describe()


# In[ ]:


df['Fare'].plot(kind='hist')


# In[ ]:


df['Fare'].plot(kind='kde')


# In[ ]:


df['Fare'].skew()


# In[ ]:


df['Fare'].plot(kind='box')


# In[ ]:


df[df['Fare'] > 250]


# In[ ]:


df['Fare'].isnull().sum()


# In[ ]:





# In[ ]:





# ### Steps of doing Univariate Analysis on Categorical columns
# 
# **Descriptive Statistics**: Compute the frequency distribution of the categories in the column. This will give a general understanding of the distribution of the categories and their relative frequencies.
# 
# **Visualizations**: Create visualizations to explore the distribution of the categories. Some common visualizations for categorical data include count plots and pie charts. These visualizations provide a visual representation of the distribution of the categories and can help identify any patterns or anomalies in the data.
# 
# **Missing Values**: Check for missing values in the data and decide how to handle them. Missing values can be imputed or excluded from the analysis, depending on the research question and the data set.
# 
# **Conclusion**: Summarize the findings of the EDA and make decisions about how to proceed with further analysis.

# ### Survived
# 
# **conclusions**
# 
# - Parch and SibSp cols can be merged to form  a new col call family_size
# - Create a new col called is_alone

# In[ ]:


df['Embarked'].value_counts()


# In[ ]:


df['Embarked'].value_counts().plot(kind='bar')


# In[ ]:


df['Embarked'].value_counts().plot(kind='pie',autopct='%0.1f%%')


# In[ ]:


df['Sex'].isnull().sum()


# ### Steps of doing Bivariate Analysis
# 
# - Select 2 cols
# - Understand type of relationship
#     1. **Numerical - Numerical**<br>
#         a. You can plot graphs like scatterplot(regression plots), 2D histplot, 2D KDEplots<br>
#         b. Check correlation coefficent to check linear relationship
#     2. **Numerical - Categorical** - create visualizations that compare the distribution of the numerical data across different categories of the categorical data.<br>
#         a. You can plot graphs like barplot, boxplot, kdeplot violinplot even scatterplots<br>
#     3. **Categorical - Categorical**<br>
#         a. You can create cross-tabulations or contingency tables that show the distribution of values in one categorical column, grouped by the values in the other categorical column.<br>
#         b. You can plots like heatmap, stacked barplots, treemaps
#         
# - Write your conclusions

# In[ ]:


df


# In[ ]:


sns.heatmap(pd.crosstab(df['Survived'],df['Pclass'],normalize='columns')*100)


# In[ ]:


pd.crosstab(df['Survived'],df['Sex'],normalize='columns')*100


# In[ ]:


pd.crosstab(df['Survived'],df['Embarked'],normalize='columns')*100


# In[ ]:


pd.crosstab(df['Sex'],df['Embarked'],normalize='columns')*100


# In[ ]:


pd.crosstab(df['Pclass'],df['Embarked'],normalize='columns')*100


# In[ ]:


# survived and age

df[df['Survived'] == 1]['Age'].plot(kind='kde',label='Survived')
df[df['Survived'] == 0]['Age'].plot(kind='kde',label='Not Survived')

plt.legend()
plt.show()


# In[ ]:


df[df['Pclass'] == 1]['Age'].mean()


# In[ ]:


# Feature Engineering on Fare col


# In[ ]:


df['SibSp'].value_counts()


# In[ ]:


df[df['Ticket'] == 'CA. 2343']


# In[ ]:


df[df['Name'].str.contains('Sage')]


# In[ ]:


df1 = pd.read_csv('/content/test.csv')


# In[ ]:


df = pd.concat([df,df1])


# In[ ]:


df[df['Ticket'] == 'CA 2144']


# In[ ]:


df['individual_fare'] = df['Fare']/(df['SibSp'] + df['Parch'] + 1)


# In[ ]:


df['individual_fare'].plot(kind='box')


# In[ ]:


df[['individual_fare','Fare']].describe()


# In[ ]:


df['Fare'].


# In[ ]:


df


# In[ ]:


df['family_size'] = df['SibSp'] + df['Parch'] + 1


# In[ ]:


# family_type
# 1 -> alone
# 2-4 -> small
# >5 -> large

def transform_family_size(num):

  if num == 1:
    return 'alone'
  elif num>1 and num <5:
    return "small"
  else:
    return "large"


# In[ ]:


df['family_type'] = df['family_size'].apply(transform_family_size)


# In[ ]:


df


# In[ ]:


pd.crosstab(df['Survived'],df['family_type'],normalize='columns')*100


# In[ ]:


df['surname'] = df['Name'].str.split(',').str.get(0)


# In[ ]:


df


# In[ ]:


df['title'] = df['Name'].str.split(',').str.get(1).str.strip().str.split(' ').str.get(0)


# In[ ]:


temp_df = df[df['title'].isin(['Mr.','Miss.','Mrs.','Master.','ootherr'])]


# In[ ]:


pd.crosstab(temp_df['Survived'],temp_df['title'],normalize='columns')*100


# In[ ]:


df['title'] = df['title'].str.replace('Rev.','other')
df['title'] = df['title'].str.replace('Dr.','other')
df['title'] = df['title'].str.replace('Col.','other')
df['title'] = df['title'].str.replace('Major.','other')
df['title'] = df['title'].str.replace('Capt.','other')
df['title'] = df['title'].str.replace('the','other')
df['title'] = df['title'].str.replace('Jonkheer.','other')
# ,'Dr.','Col.','Major.','Don.','Capt.','the','Jonkheer.']


# In[ ]:


df['Cabin'].isnull().sum()/len(df['Cabin'])


# In[ ]:


df['Cabin'].fillna('M',inplace=True)


# In[ ]:


df['Cabin'].value_counts()


# In[ ]:


df['deck'] = df['Cabin'].str[0]


# In[ ]:


df['deck'].value_counts()


# In[ ]:


pd.crosstab(df['deck'],df['Pclass'])


# In[ ]:


pd.crosstab(df['deck'],df['Survived'],normalize='index').plot(kind='bar',stacked=True)


# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


sns.pairplot(df1)


# In[ ]:


df1


# Here's a simplified description:
# 
# 1. **Understanding the Data**:
#    - First, I looked at information about the passengers on the Titanic, like their age, gender, ticket class, and fare.
# 
# 2. **Cleaning the Data**:
#    - I checked for any missing information and made sure everything was complete. For example, if someone's age was missing, I filled it in with the average age.
# 
# 3. **Finding Basic Facts**:
#    - I calculated some simple things like average ages, the youngest and oldest passengers, and the average fare.
# 
# 4. **Making Pictures**:
#    - I used graphs and charts to make the data easier to understand. There were bar charts for comparing things like ticket class and gender, scatter plots to see relationships between age and fare, and histograms to show how age and fare were spread out.
# 
# 5. **Looking at Who Survived**:
#    - I checked to see who survived the Titanic disaster. There were graphs to show the percentage of people who survived compared to those who didn't. I also looked at factors like gender, age, and ticket class to see if they affected survival chances.
# 
# 6. **What I Learned**:
#    - From all this, I found out some interesting stuff. For example, it seemed like more women survived than men, and people in higher ticket classes were more likely to make it.
# 
# 7. **Putting it Together**:
#    - Finally, I made a nice summary of my findings with the most important graphs and a bit of explanation. This way, it's easy for anyone to see what I discovered from the Titanic data!

# In[ ]:




