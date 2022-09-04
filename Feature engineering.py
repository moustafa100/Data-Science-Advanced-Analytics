#!/usr/bin/env python
# coding: utf-8

# <h1 style='text-align:center;'> Feature engineering </h1>

# ### 0. Import packages
# 
# 
# Load the necessary packages for this exercise

# In[1]:


import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns


# In[2]:


# Show plots in jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Set plot style
sns.set(color_codes=True)


# In[4]:


# Set maximum number of columns to be displayed
pd.set_option('display.max_columns', 100)


# ### 0. Loading data
# 
# #### Data directory
# 
# Explicitly show how paths are indicated

# In[5]:


cd "C:/Users/top/Desktop/BCGprojects"


# In[6]:


PICKLE_TRAIN_DIR = os.path.join("C:/Users/top/Desktop/BCGprojects", "train_data.pkl")
PICKLE_HISTORY_DIR = os.path.join("C:/Users/top/Desktop/BCGprojects", "history_data.pkl")


# #### Load data into dataframes
# 
# Data file are in csv format, hence we can use the built in functions in pandas

# In[7]:


history_data = pd.read_pickle(PICKLE_HISTORY_DIR)
train = pd.read_pickle(PICKLE_TRAIN_DIR)


# ### 1. Feature engineering
# 
# Since we have the consumption data for each of the companies for the year 2015, we will create new features using the average of the year, the
# last six months, and the last three months to our model.
# 
# 

# In[8]:


mean_year = history_data.groupby(["id"]).mean().reset_index()


# In[9]:


mean_6m = history_data[history_data["price_date"] > "2015-06-01"].groupby(["id"]).mean().reset_index()


# In[10]:


mean_3m = history_data[history_data["price_date"] > "2015-10-01"].groupby(["id"]).mean().reset_index()


# In[11]:


### Combine them in a single dataframe
mean_year = mean_year.rename(index=str, columns={"price_p1_var": "mean_year_price_p1_var",
                                                    "price_p2_var": "mean_year_price_p2_var",
                                                    "price_p3_var": "mean_year_price_p3_var",
                                                    "price_p1_fix": "mean_year_price_p1_fix",
                                                    "price_p2_fix": "mean_year_price_p2_fix",
                                                    "price_p3_fix": "mean_year_price_p3_fix",})
mean_year["mean_year_price_p1"] = mean_year["mean_year_price_p1_var"] + mean_year["mean_year_price_p1_fix"]
mean_year["mean_year_price_p2"] = mean_year["mean_year_price_p2_var"] + mean_year["mean_year_price_p2_fix"]
mean_year["mean_year_price_p3"] = mean_year["mean_year_price_p3_var"] + mean_year["mean_year_price_p3_fix"]


# In[12]:


mean_6m = mean_6m.rename(index=str, columns={"price_p1_var": "mean_6m_price_p1_var",
                                                "price_p2_var": "mean_6m_price_p2_var",
                                                "price_p3_var": "mean_6m_price_p3_var",
                                                "price_p1_fix": "mean_6m_price_p1_fix",
                                                "price_p2_fix": "mean_6m_price_p2_fix",
                                                "price_p3_fix": "mean_6m_price_p3_fix",})
mean_6m["mean_6m_price_p1"] = mean_6m["mean_6m_price_p1_var"] + mean_6m["mean_6m_price_p1_fix"]
mean_6m["mean_6m_price_p2"] = mean_6m["mean_6m_price_p2_var"] + mean_6m["mean_6m_price_p2_fix"]
mean_6m["mean_6m_price_p3"] = mean_6m["mean_6m_price_p3_var"] + mean_6m["mean_6m_price_p3_fix"]


# In[13]:


mean_3m = mean_3m.rename(index=str, columns={"price_p1_var": "mean_3m_price_p1_var",
                                                "price_p2_var": "mean_3m_price_p2_var",
                                                "price_p3_var": "mean_3m_price_p3_var",
                                                "price_p1_fix": "mean_3m_price_p1_fix",
                                                "price_p2_fix": "mean_3m_price_p2_fix",
                                                "price_p3_fix": "mean_3m_price_p3_fix",})
mean_3m["mean_3m_price_p1"] = mean_3m["mean_3m_price_p1_var"] + mean_3m["mean_3m_price_p1_fix"]
mean_3m["mean_3m_price_p2"] = mean_3m["mean_3m_price_p2_var"] + mean_3m["mean_3m_price_p2_fix"]
mean_3m["mean_3m_price_p3"] = mean_3m["mean_3m_price_p3_var"] + mean_3m["mean_3m_price_p3_fix"]


# Now we will merge them into a single dataframe
# 
# 
# **Note**: I am not confident the mean_6m and mean_3m could help the prediction model. We will see below the variables are also highly correlated to
# actually using only the mean_year is OK
# 

# In[14]:


#features = pd.merge(mean_year,mean_6m, on="id")
#features = pd.merge(features,mean_3m, on="id")
features = mean_year


# What if we could create a new variable that could provide us more relevant insights?
# 
# 
#                   We will define a variable tenure = date_end - date_activ

# In[15]:


train["tenure"] = ((train["date_end"]-train["date_activ"])/ np.timedelta64(1, "Y")).astype(int)


# In[16]:


tenure = train[["tenure", "churn", "id"]].groupby(["tenure", "churn"])["id"].count().unstack(level=1)
tenure


# In[17]:


tenure_percentage = (tenure.div(tenure.sum(axis=1), axis=0)*100)


# In[18]:


tenure.plot(kind="bar",
 figsize=(18,10),
 stacked=True,
rot=0,
 title= "Tenure")
# Rename legend
plt.legend(["Retention", "Churn"], loc="upper right")
# Labels
plt.ylabel("No. of companies")
plt.xlabel("No. of years")
plt.show()


# We can clearly that churn is very low for companies which joined recently or that have made the contract a long time ago. With the higher number of
# churners within the 3-7 years of tenure.
# 

# We will also transform the dates provided in such a way that we can make more sense out of those.
# 
# 
# 
#         months_activ : Number of months active until reference date (Jan 2016)
# 
#         months_to_end : Number of months of the contract left at reference date (Jan 2016)
# 
#         months_modif_prod : Number of months since last modification at reference date (Jan 2016)
# 
#         months_renewal : Number of months since last renewal at reference date (Jan 2016)
#     
#     
# To create the month column we will follow a simple process:
# 1. Substract the reference date and the column date
# 2. Convert the timedelta in months
# 3. Convert to integer (we are not interested in having decimal months)

# In[22]:


def convert_months(reference_date, dataframe, column):
    """
    Input a column with timedeltas and return months
    """
    time_delta = REFERENCE_DATE - dataframe[column]
    months = (time_delta / np.timedelta64(1, "M")).astype(int)
    return months


# In[23]:


# Create reference date as provided on the exercise statement
REFERENCE_DATE = datetime.datetime(2016,1,1)


# In[24]:


train["months_activ"] = convert_months(REFERENCE_DATE, train, "date_activ")
train["months_to_end"] = -convert_months(REFERENCE_DATE, train, "date_end")
train["months_modif_prod"] = convert_months(REFERENCE_DATE, train, "date_modif_prod")
train["months_renewal"] = convert_months(REFERENCE_DATE, train, "date_renewal")


# Let's see if we can get any insights

# In[28]:


def plot_churn_by_month(dataframe, column, fontsize_=11):
    """
     Plot churn distribution by monthly variable
    """
    temp = dataframe[[column, "churn", "id"]].groupby([column, "churn"])["id"].count().unstack(level=1)
    temp.plot(kind="bar",
    figsize=(18,10),
    stacked=True,
    rot=0,
    title= column)
    # Rename legend
    plt.legend(["Retention", "Churn"], loc="upper right")
    # Labels
    plt.ylabel("No. of companies")
    plt.xlabel("No. of months")
    # Set xlabel fontsize
    plt.xticks(fontsize=fontsize_)
    plt.show()


# In[29]:


plot_churn_by_month(train, "months_activ", 7)


# In[30]:


plot_churn_by_month(train, "months_to_end")


# In[31]:


plot_churn_by_month(train, "months_modif_prod", 8)


# In[32]:


plot_churn_by_month(train, "months_renewal")


# Remove the date columns

# In[33]:


train.drop(columns=["date_activ", "date_end", "date_modif_prod", "date_renewal"],inplace=True)


# #### Transforming boolean data
# For the column has_gas, we will replace t for True or 1 and f for False or 0 .\ This process is usually referred as onehot encoding

# In[34]:


train["has_gas"]=train["has_gas"].replace(["t", "f"],[1,0])


# ### Categorical data and dummy variables
# When training our model we cannot use string data as such, so we will need to encode it into numerical data. The easiest method is mapping
# each category to an integer ( label encoding ) but this will not work because the model will misunderstand the data to be in some kind of order or
# hierarchy, 0 < 1 < 2 < 3 ...
# 
# 
# For that reason we will use a method with dummy variables or onehot encoder
# 
# 
#    ##### Categorical data channel_sales
# What we are doing here relatively simple, we want to convert each category into a new dummy variable which will have 0 s and 1 s depending
# whether than entry belongs to that particular category or not
# 
# 
# First of all let's replace the Nan values with a string called null_values_channel
# 

# In[35]:


train["channel_sales"] = train["channel_sales"].fillna("null_values_channel")


# Now transform the channel_sales column into categorical data type

# In[36]:


# Transform to categorical data type
train["channel_sales"] = train["channel_sales"].astype("category")


# We want to see how many categories we will end up with

# In[37]:


pd.DataFrame({"Samples in category": train["channel_sales"].value_counts()})


# So that means we will create 8 different dummy variables . Each variable will become a different column.

# In[38]:


# Create dummy variables
categories_channel = pd.get_dummies(train["channel_sales"], prefix = "channel")


# In[39]:


# Rename columns for simplicity
categories_channel.columns = [col_name[:11] for col_name in categories_channel.columns]


# In[40]:


categories_channel.head(5)


# We will explain the concept of **multicollinearity** in the next section. Simply put, multicollinearity is when two or more independent variables in a
# regression are highly related to one another, such that they do not provide unique or independent information to the regression.\
# **Multicollinearity** can affect our models so we will remove one of the columns.
# 

# In[41]:


categories_channel.drop(columns=["channel_nul"],inplace=True)


# #### Categorical data origin_up
# 
# 
# First of all let's replace the Nan values with a string called null_values_origin
# 

# In[42]:


train["origin_up"] = train["origin_up"].fillna("null_values_origin")


# Now transform the origin_up column into categorical data type

# In[44]:


train["origin_up"] = train["origin_up"].astype("category")


# We want to see how many categories we will end up with

# In[46]:


pd.DataFrame({"Samples in category": train["origin_up"].value_counts()})


# So that means we will create 8 different dummy variables . Each variable will become a different column.

# In[47]:


# Create dummy variables
categories_origin = pd.get_dummies(train["origin_up"], prefix = "origin")
# Rename columns for simplicity
categories_origin.columns = [col_name[:10] for col_name in categories_origin.columns]


# In[48]:


categories_origin.head(5)


# In[49]:


categories_origin.drop(columns=["origin_nul"],inplace=True)


# #### Categorical data - values_activity
# First of all let's replace the Nan values with a string called null_values_activity

# In[50]:


train["activity_new"] = train["activity_new"].fillna("null_values_activity")


# We want to see how many categories we will end up with

# In[52]:


categories_activity = pd.DataFrame({"Activity samples":train["activity_new"].value_counts()})
categories_activity


# As we can see below there are too many categories with very few number of samples. So we will replace any category with less than 75 samples as 
# null_values_category

# In[53]:


# Get the categories with less than 75 samples
to_replace = list(categories_activity[categories_activity["Activity samples"] <= 75].index)
# Replace them with `null_values_categories`
train["activity_new"]=train["activity_new"].replace(to_replace,"null_values_activity")


# In[54]:


# Create dummy variables
categories_activity = pd.get_dummies(train["activity_new"], prefix = "activity")
# Rename columns for simplicity
categories_activity.columns = [col_name[:12] for col_name in categories_activity.columns]


# In[55]:


categories_activity.head(5)


# In[56]:


categories_activity.drop(columns=["activity_nul"],inplace=True)


# #### Merge dummy variables to main dataframe
# We will merge all the new categories into our main dataframe and remove the old categorical columns
# 

# In[57]:


# Use common index to merge
train = pd.merge(train, categories_channel, left_index=True, right_index=True)
train = pd.merge(train, categories_origin, left_index=True, right_index=True)
train = pd.merge(train, categories_activity, left_index=True, right_index=True)


# In[58]:


train.drop(columns=["channel_sales", "origin_up", "activity_new"],inplace=True)


# ### Log transformation
# Remember from the previous exercise that a lot of the variables we are dealing with are highly skewed to the right.\
# **Why is skewness relevant?** Skewness is not "bad" per se. Nonetheless, some predective models make fundamental assumptions related to
# variables being "normally distributed". Hence, the model will perform poorly if the data is highly skewed.
# There are several methods in which we can reduce skewness such as square root , cube root , and log . In this case, we will use a log
# transformation which is usually recommended for right skewed data.
# 

# In[59]:


train.describe()


# Particularly relevant to look at the standard deviation std which is very very high for some variables.
# 
# Log transformation does not work with negative data, so we will convert the negative values to NaN .
# 

# Also we cannot apply a log transformation to 0 valued entries, so we will add a constant 1

# In[61]:


# Remove negative values
train.loc[train.cons_12m < 0,"cons_12m"] = np.nan
train.loc[train.cons_gas_12m < 0,"cons_gas_12m"] = np.nan
train.loc[train.cons_last_month < 0,"cons_last_month"] = np.nan
train.loc[train.forecast_cons_12m < 0,"forecast_cons_12m"] = np.nan
train.loc[train.forecast_cons_year < 0,"forecast_cons_year"] = np.nan
train.loc[train.forecast_meter_rent_12m < 0,"forecast_meter_rent_12m"] = np.nan
train.loc[train.imp_cons < 0,"imp_cons"] = np.nan


# In[62]:


# Apply log10 transformation
train["cons_12m"] = np.log10(train["cons_12m"]+1)
train["cons_gas_12m"] = np.log10(train["cons_gas_12m"]+1)
train["cons_last_month"] = np.log10(train["cons_last_month"]+1)
train["forecast_cons_12m"] = np.log10(train["forecast_cons_12m"]+1)
train["forecast_cons_year"] = np.log10(train["forecast_cons_year"]+1)
train["forecast_meter_rent_12m"] = np.log10(train["forecast_meter_rent_12m"]+1)
train["imp_cons"] = np.log10(train["imp_cons"]+1)


# Now let's see how the distribution looks like.

# In[63]:


fig, axs = plt.subplots(nrows=7, figsize=(18,50))
# Plot histograms
sns.distplot((train["cons_12m"].dropna()), ax=axs[0])
sns.distplot((train[train["has_gas"]==1]["cons_gas_12m"].dropna()), ax=axs[1])
sns.distplot((train["cons_last_month"].dropna()), ax=axs[2])
sns.distplot((train["forecast_cons_12m"].dropna()), ax=axs[3])
sns.distplot((train["forecast_cons_year"].dropna()), ax=axs[4])
sns.distplot((train["forecast_meter_rent_12m"].dropna()), ax=axs[5])
sns.distplot((train["imp_cons"].dropna()), ax=axs[6])
plt.show()


# In[64]:


fig, axs = plt.subplots(nrows=7, figsize=(18,50))
# Plot boxplots
sns.boxplot((train["cons_12m"].dropna()), ax=axs[0])
sns.boxplot((train[train["has_gas"]==1]["cons_gas_12m"].dropna()), ax=axs[1])
sns.boxplot((train["cons_last_month"].dropna()), ax=axs[2])
sns.boxplot((train["forecast_cons_12m"].dropna()), ax=axs[3])
sns.boxplot((train["forecast_cons_year"].dropna()), ax=axs[4])
sns.boxplot((train["forecast_meter_rent_12m"].dropna()), ax=axs[5])
sns.boxplot((train["imp_cons"].dropna()), ax=axs[6])
plt.show()


# In[65]:


train.describe()


# The distributions look much closer to normal distributions now!
# 
# Notice how the standard deviation std has changed.
# 
# From the boxplots we can still see some values are quite far from the range ( outliers ). We will deal with them later.
# 

# ### 2. High correlation variables
# 
# 
# Calculate the correlation of the variables

# In[66]:


# Calculate correlation of variables
correlation = features.corr()


# In[67]:


# Plot correlation
plt.figure(figsize=(19,15))
sns.heatmap(correlation, xticklabels=correlation.columns.values,
 yticklabels=correlation.columns.values, annot = True, annot_kws={'size':10})
# Axis ticks size
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# We can remove highly correlated variables.
# 
# 
# Multicollinearity happens when one predictor variable in a multiple regression model can be linearly predicted from the others with a high degree of
# accuracy. This can lead to skewed or misleading results. Luckily, decision trees and boosted trees algorithms are immune to multicollinearity by
# nature. When they decide to split, the tree will choose only one of the perfectly correlated features. However, other algorithms like Logistic Regression
# or Linear Regression are not immune to that problem and should be fixed before training the model.
# 

# In[69]:


# Calculate correlation of variables
correlation = train.corr()


# In[70]:


# Plot correlation
plt.figure(figsize=(20,18))
sns.heatmap(correlation, xticklabels=correlation.columns.values,
 yticklabels=correlation.columns.values, annot = True, annot_kws={'size':10})
# Axis ticks size
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# As expected, num_years_antig has a high correlation with months_activ (it provides us the same information).
# 
# We can remove variables with very high correlation.
# 

# In[71]:


train.drop(columns=["num_years_antig", "forecast_cons_year"],inplace=True)


# ### 3. Removing outliers
# As we identified during the exploratory phase, the consumption data has several outliers. We are going to remove those outliers\
# **What are the criteria to identify an outlier?**
# 
# The most common way to identify an outlier are:
# 
# 1. Data point that falls outside of 1.5 times of an interquartile range above the 3rd quartile and below the 1st quartile
# 
# OR
# 
# 2. Data point that falls outside of 3 standard deviations.
# 
# Once, we have identified the outlier,**What do we do with the outliers?**\
# There are several ways to handle with those outliers such as removing them (this works well for massive datasets) or replacing them with sensible data
# (works better when the dataset is not that big).
# We will replace the outliers with the mean (average of the values excluding outliers).
# 

# In[72]:


def replace_outliers_z_score(dataframe, column, Z=3):
    """
    Replace outliers with the mean values using the Z score.
    Nan values are also replaced with the mean values.
    Parameters
    ----------
    dataframe : pandas dataframe
    Contains the data where the outliers are to be found
    column : str
    Usually a string with the name of the column 

    Returns
    -------
    Dataframe
    With outliers under the lower and above the upper bound removed
    """
    from scipy.stats import zscore

    df = dataframe.copy(deep=True)
    df.dropna(inplace=True, subset=[column])

    # Calculate mean without outliers
    df["zscore"] = zscore(df[column])
    mean_ = df[(df["zscore"] > -Z) & (df["zscore"] < Z)][column].mean()

    # Replace with mean values
    dataframe[column] = dataframe[column].fillna(mean_)
    dataframe["zscore"] = zscore(dataframe[column])
    no_outliers = dataframe[(dataframe["zscore"] < -Z) | (dataframe["zscore"] > Z)].shape[0]
    dataframe.loc[(dataframe["zscore"] < -Z) | (dataframe["zscore"] > Z),column] = mean_

    # Print message
    print("Replaced:", no_outliers, " outliers in ", column)
    return dataframe.drop(columns="zscore")


# In[73]:


for c in features.columns:
    if c != "id":
         features = replace_outliers_z_score(features,c)


# In[74]:


features.reset_index(drop=True, inplace=True)


# In[75]:


for c in train.columns:
     if c != "id":
        train = replace_outliers_z_score(train,c)


# In[76]:


train.reset_index(drop=True, inplace=True)


# In[77]:


fig, axs = plt.subplots(nrows=7, figsize=(18,50))
# Plot boxplots
sns.boxplot((train["cons_12m"].dropna()), ax=axs[0])
sns.boxplot((train[train["has_gas"]==1]["cons_gas_12m"].dropna()), ax=axs[1])
sns.boxplot((train["cons_last_month"].dropna()), ax=axs[2])
sns.boxplot((train["forecast_cons_12m"].dropna()), ax=axs[3])
#sns.boxplot((train["forecast_cons_year"].dropna()), ax=axs[4])
sns.boxplot((train["forecast_meter_rent_12m"].dropna()), ax=axs[5])
sns.boxplot((train["imp_cons"].dropna()), ax=axs[6])
plt.show()


# ### 4. Pickling
# 
# 
# We will pickle the data so that we can easily retrieve it in for the next exercise.
# 

# In[82]:


PICKLE_TRAIN_DIR = os.path.join("C:/Users/top/Desktop/BCGprojects", "train_data.pkl")
PICKLE_HISTORY_DIR = os.path.join("C:/Users/top/Desktop/BCGprojects", "history_data.pkl")


# In[83]:


pd.to_pickle(train, PICKLE_TRAIN_DIR)
pd.to_pickle(history_data, PICKLE_HISTORY_DIR)


# In[ ]:




