#!/usr/bin/env python
# coding: utf-8

# <h1 style='text-align:center;'> Exploratory Data Analysis </h1>

# ### Sub-Task 1:
# 
# Perform some exploratory data analysis. Look into the data types, data statistics, specific parameters, and variable distributions. This first subtask is for you to gain a holistic understanding of the dataset.

# 1. Loading data through pandas
#         A. Pandas built-in functions
#             B. Printing to the screen specific areas of a dataframe
#                 C. Combining two dataframes
# 2. General statistics of a dataframe
#         A. Data types
#             B. Data statistics
#                 C. Missing data
# 3. Data visualization
#         A. Deep diving in specific parameters
#             B. Visualising variable distributions

# ## 0. Import packages
# Load the necessary packages
# 

# In[1]:


import matplotlib as plt
import pandas  as pd
import numpy as np
import seaborn as sns
import datetime
import os
import pickle


# In[2]:


# Show plots in jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Set plot style
sns.set(color_codes=True)


# In[4]:


# Set maximum number of columns to be displayed
pd.set_option('display.max_columns', 100)


# ## 1. Loading data

# In[5]:


train_data=pd.read_csv("ml_case_training_data.csv")
churn_data=pd.read_csv("ml_case_training_output.csv")
history_data=pd.read_csv("ml_case_training_hist_data.csv")


# ### Load data into a dataframe
# Data file are in csv format, hence we can use the built-in functions form pandas pd.read_csv(filename)

# In[6]:


train_data.head(20)


# In[7]:


churn_data.head()


# In[8]:


history_data.head()


# #### Combining two dataframes
# We loaded data in two different pandas dataframes. Nonetheless, we might be interested in putting the data into a single dataframe to access it more
# easily. We can merge the two dataframes on a common column ( id ) using the function pd.merge() from pandas.
# 

# In[9]:


train=pd.merge(train_data,churn_data,on="id")


# In[72]:


train.head(5)


# ### 2. General statistics of a dataframe
# 

# #### Data types
# Often, it is useful to understand what data we are dealing with, as the data types might end up causing errors into our analysis at a later stage.
# Below, we can quickly see the dates in our dataset are not datetime types yet, which means we might need to convert them. In addition, we can see that the churn is full of integers so we can keep it in that form.
# 
# **Note**: We've transformed the output to a dataframe to facilitate visualization

# In[11]:


pd.DataFrame({"Data type":train.dtypes})


# In[12]:


pd.DataFrame({"Data type":history_data.dtypes})


# In[13]:


train.describe()


# from above we can obtain a lot of information about the dataset we are dealing with. Some key facts include:

# 1. The minimum consumption and forecasts for electricity and gas (yearly and monthly) are negative. This could mean that
# the client companies are producing energy and therefore energy should be "returned", although it is unlikely and we will consider it
# as corrupted data.
# 2. The campaign_disc_ele is an empty column. We verify it by running\
# train["campaign_disc_ele"].isnull().values.all()
# 3. Highly skewed data when we look at the percentiles.
# 

# In[14]:


history_data.describe()


# #### Missing data
# 

# We are also concerned we have a lot of missing data so we can check how much of our data is missing.\
# **Note**: We've transformed the output to a dataframe to facilitate visualization. This will be used for data cleaning in the next exercise.
# 

# In[15]:


pd.DataFrame({"Missing value%":train.isnull().sum()/len(train.index)*100})


# Some of the columns might need to be removed since they have more than 75% of the data missing

# In[16]:


pd.DataFrame({"Missing value":history_data.isnull().sum()/len(history_data.index)*100})


# In this case the missing data is very small, so we might be able to easily replace the missing values with approximations

# ###  3. Data visualization

# #### Churn
# Let's see the churning rate
# 

# In[17]:


churn=train[["id","churn"]]


# In[18]:


# Rename columns for visualization purposes
churn.columns=["companies","churn"]


# In[19]:


def plot_stacked_bars(dataframe, title_, size_=(18, 10), rot_=0, legend_="lower right"):
    """
    Plot stacked bars with annotations
    """
    ax = dataframe.plot(kind="bar",
                      stacked=True,
                      figsize=size_,
                          rot=rot_,
                        title=title_)
    # Annotate bars
    annotate_stacked_bars(ax, textsize=14)
    # Rename legend
    plt.pyplot.legend(["Retention", "Churn"],loc=legend_)
    # Labels
    plt.pyplot.ylabel("Company base (%)")
    plt.pyplot.show()
def annotate_stacked_bars(ax, pad=0.99, colour="white", textsize=13):
    """
    Add value annotations to the bars
    """
    # Iterate over the plotted rectanges/bars
    for p in ax.patches:
        # Calculate annotation
        value = str(round(p.get_height(),1))
        # If value is 0 do not annotate
        if value == '0.0':
               continue
        ax.annotate(value,
           ((p.get_x()+ p.get_width()/2)*pad-0.05, (p.get_y()+p.get_height()/2)*pad),
                        color=colour,
                              size=textsize,
                                          )
   


# In[20]:


churn_total=churn.groupby(churn["churn"]).count()
churn_precentage=churn_total/churn_total.sum()*100


# In[21]:


plot_stacked_bars(churn_precentage.transpose(),"Churning status", (5,5), legend_="lower right")


# #### SME activity

# Let's show the activity distribution of the companies as well as the sales channel.\
# Intuitively this might be an important predictive feature for energy consumption

# In[22]:


activity = train[["id","activity_new","churn"]]


# In[23]:


activity = activity.groupby([activity["activity_new"],
 activity["churn"]])["id"].count().unstack(level=1).sort_values(by=[0], ascending=False)
activity


# In[24]:


activity.plot(kind="bar"
                ,figsize=(18,10)
             ,stacked=True
             ,width=2,title="SME Companies")
plt.pyplot.xlabel("Activity")
plt.pyplot.ylabel("Number of Campanies")
plt.pyplot.legend(["Retention","churn"],loc="upper right")
plt.pyplot.xticks([])
plt.pyplot.show()


# The distribution of the classes over the labeled data despite the lack of 60% of the entries.\
# We see churn is not specifically related to any SME cateogry in particular.\
# **Note**: Not showing the labels in the x-axis to facilitate visualization\
# If we take a look at the values percentage-wise
# 

# In[25]:


activity_total=activity.fillna(0)[0]+activity.fillna(0)[1]
activity_total


# In[26]:


activity_total=activity.fillna(0)[0]+activity.fillna(0)[1]
activity_percentage=activity.fillna(0)[1]/(activity_total)*100
pd.DataFrame({"percentage churn":activity_percentage,"Total Companies":activity_total}).sort_values(by="percentage churn",
                                                                                                   ascending=False).head(10)


# If sorted by activity some companies have churned a 100% but this is due to the fact that only a few companies belong to that activity.

# **How will the SME activity influence our predictive model?**\
# Our predictive model is likely to struggle accurately predicting the the SME activity due to the large number of categories and low
# number of companies belonging to each category.

# #### Sales channel

# The sales channel seems to be an important feature when predecting the churning of a user. It is not the same if the sales were through email or
# telephone.
# 

# In[27]:


channel=train[["id","channel_sales","churn"]]
channel=channel.groupby([channel["channel_sales"],channel["churn"]])["id"].count().unstack(level=1).fillna(0)
channel


# In[28]:


channel_churn=(channel.div(channel.sum(axis=1),axis=0)*100).sort_values(by=[1],ascending=False)


# In[29]:


plot_stacked_bars(channel_churn,"Sales Channel", rot_=30)


# In[30]:


channel_total=channel.fillna(0)[0]+channel.fillna(0)[1]
channel_percentage=channel.fillna(0)[1]/(channel_total)*100
pd.DataFrame({"churn percentage":channel_percentage,"Total Companies":channel_total}).sort_values(by="churn percentage",
                                                                                                 ascending=False).head(10)


# #### Consumption
# 

# Let's see the distribution of the consumption over the last year and last month
# 

# In[31]:


consumption=train[["id","cons_12m", "cons_gas_12m","cons_last_month", "imp_cons", "has_gas", "churn"]]


# The most straight forward to visualise and identify the distribution of uni-variate data is through histograms

# In[32]:


def plot_distribution(dataframe,column,ax,bins_=50):
    
    """
    Plot variable distirbution in a stacked histogram of churned or retained company
    """
    temp = pd.DataFrame({"Retention":dataframe[dataframe["churn"]==0][column],
                                      "Churn":dataframe[dataframe["churn"]==1][column]})
    temp[["Retention","Churn"]].plot(kind='hist',bins=bins_,ax=ax,stacked=True)
    ax.set_xlabel(column)
    ax.ticklabel_format(style='plain', axis='x')


# In[33]:


fig, axs = plt.pyplot.subplots(nrows=4, figsize=(18,25))
plot_distribution(consumption, "cons_12m", axs[0])
# Note that the gas consumption must have gas contract
plot_distribution(consumption[consumption["has_gas"] == "t"], "cons_gas_12m", axs[1])
plot_distribution(consumption, "cons_last_month", axs[2])
plot_distribution(consumption, "imp_cons", axs[3])


# We can clearly see in here that the consumption data is highly skewed to the right, presenting a very long right-tail towards the higher values of the
# distribution.
# 
# 
# The values on the higher end and lower ends of the distribution are likely to be outliers. We can use a standard plot to visualise the outliers in more
# detail. A boxplot is a standardized way of displaying the distribution of data based on a five number summary (“minimum”, first quartile (Q1), median,
# third quartile (Q3), and “maximum”). It can tell us about our outliers and what their values are. It can also tell us if our data is symmetrical, how tightly
# our data is grouped, and if and how our data is skewed.
# 

# In[34]:


fig, axs = plt.pyplot.subplots(nrows=4, figsize=(18,25))
sns.boxplot(consumption["cons_12m"], ax=axs[0])
sns.boxplot(consumption[consumption["has_gas"] == "t"]["cons_gas_12m"], ax=axs[1])
sns.boxplot(consumption["cons_last_month"], ax=axs[2])
sns.boxplot(consumption["imp_cons"], ax=axs[3])
for ax in axs:
    ax.ticklabel_format(style='plain', axis='x')
axs[0].set_xlim(-200000, 2000000)
axs[1].set_xlim(-200000, 2000000)
axs[2].set_xlim(-20000, 100000)
plt.pyplot.show()


# It is very clear now that we have a highly skewed distribution, and several outliers.

# #### Dates
# 

# In[35]:


dates=train[["id","date_activ","date_end", "date_modif_prod","date_renewal","churn"]].copy()


# In[36]:


# Transform date columns to datetime type
dates["date_activ"]=pd.to_datetime(dates["date_activ"],format='%Y-%m-%d')
dates["date_end"]=pd.to_datetime(dates["date_end"],format='%Y-%m-%d')
dates["date_modif_prod"]=pd.to_datetime(dates["date_modif_prod"],format='%Y-%m-%d')
dates["date_renewal"]=pd.to_datetime(dates["date_renewal"],format='%Y-%m-%d')


# In[37]:


def plot_date(dataframe,column,fontsize_=12):
    """
    Plot monthly churn and retention distribution
    """
    temp=dataframe[[column,"churn","id"]].set_index(column).groupby([pd.Grouper(freq='m'),"churn"]).count().unstack(level=1)
    ax=temp.plot(kind='bar',stacked=True,figsize=(18,10),rot=0)
    ax.set_xticklabels(map(lambda x: line_format(x), temp.index))
    plt.pyplot.xticks(fontsize=fontsize_)
    plt.pyplot.ylabel("Number of companies")
    plt.pyplot.legend(["Retention", "Churn"], loc="upper right")
    plt.pyplot.show()
def line_format(label):
    """
    Convert time label to the format of pandas line plot
    """
    month=label.month_name()[:1]
    if label.month_name() == "January":
        month += f'\n{label.year}'
    return month


# In[38]:


plot_date(dates,"date_activ",fontsize_=8)


# In[39]:


plot_date(dates,"date_end")


# In[40]:


plot_date(dates, "date_modif_prod", fontsize_=8)


# In[41]:


plot_date(dates, "date_renewal")


# As a remark in here, we can visualize the distribution of churned companies according to the date. However, this does not provide us with any useful
# insight. We will create a new feature using the raw dates

# #### Forecast

# In[42]:


forcast=train[["id","forecast_base_bill_ele","forecast_base_bill_year",
 "forecast_bill_12m","forecast_cons","forecast_cons_12m",
 "forecast_cons_year","forecast_discount_energy","forecast_meter_rent_12m",
 "forecast_price_energy_p1","forecast_price_energy_p2",
 "forecast_price_pow_p1","churn"]]


# In[43]:


fig, axs = plt.pyplot.subplots(nrows=11, figsize=(18,50))
plot_distribution(forcast, "forecast_base_bill_ele", axs[0])
plot_distribution(forcast, "forecast_base_bill_year", axs[1])
plot_distribution(forcast, "forecast_bill_12m", axs[2])
plot_distribution(forcast, "forecast_cons", axs[3])
plot_distribution(forcast, "forecast_cons_12m", axs[4])
plot_distribution(forcast, "forecast_cons_year", axs[5])
plot_distribution(forcast, "forecast_discount_energy", axs[6])
plot_distribution(forcast, "forecast_meter_rent_12m", axs[7])
plot_distribution(forcast, "forecast_price_energy_p1", axs[8])
plot_distribution(forcast, "forecast_price_energy_p2", axs[9])
plot_distribution(forcast, "forecast_price_pow_p1", axs[10])


# Similarly to the consumption plots, we can observe that a lot of the variables are highly skewed to the right,creating a very long tail on the higher
# values.

# #### Contract type (electricity, gas)
# 

# In[44]:


contract_type=train[["id","has_gas","churn"]]


# In[45]:


contract=contract_type.groupby([contract_type["churn"],contract_type["has_gas"]])["id"].count().unstack(level=0)


# In[46]:


contract


# In[47]:


contract_percentage = (contract.div(contract.sum(axis=1), axis=0)*100).sort_values(by=[1], ascending=False)


# In[48]:


plot_stacked_bars(contract_percentage, "Contract type (with gas)")


# #### Margins

# In[49]:


margin = train[["id","margin_gross_pow_ele","margin_net_pow_ele","net_margin"]]


# In[50]:


fig, axs = plt.pyplot.subplots(nrows=3, figsize=(18,20))

sns.boxplot(margin["margin_gross_pow_ele"], ax=axs[0])
sns.boxplot(margin["margin_net_pow_ele"],ax=axs[1])
sns.boxplot(margin["net_margin"], ax=axs[2])
axs[0].ticklabel_format(style='plain', axis='x')
axs[1].ticklabel_format(style='plain', axis='x')
axs[2].ticklabel_format(style='plain', axis='x')
plt.pyplot.show()


# We can observe a few outliers in here as well.

# #### Subscribed power

# In[51]:


power = train[["id","pow_max", "churn"]].fillna(0)


# In[52]:


fig, axs = plt.pyplot.subplots(nrows=1, figsize=(18,10))
plot_distribution(power, "pow_max", axs)


# #### Others

# In[53]:


others = train[["id","nb_prod_act","num_years_antig", "origin_up", "churn"]]


# In[54]:


products = others.groupby([others["nb_prod_act"],others["churn"]])["id"].count().unstack(level=1)
products_percentage = (products.div(products.sum(axis=1), axis=0)*100).sort_values(by=[1], ascending=False)
plot_stacked_bars(products_percentage, "Number of products")


# In[55]:


years_antig = others.groupby([others["num_years_antig"],others["churn"]])["id"].count().unstack(level=1)
years_antig_percentage = (years_antig.div(years_antig.sum(axis=1), axis=0)*100)
plot_stacked_bars(years_antig_percentage, "Number years")


# In[56]:


origin = others.groupby([others["origin_up"],others["churn"]])["id"].count().unstack(level=1)
origin_percentage = (origin.div(origin.sum(axis=1), axis=0)*100)
plot_stacked_bars(origin_percentage, "Origin contract/offer")


# ### 4.Data cleaning

# #### Missing data

# In[57]:


train.head(15)


# In[58]:


(train.isnull().sum()/len(train.index)*100).plot(kind="bar",figsize=(18,10))
plt.pyplot.xlabel("variables")
plt.pyplot.ylabel("missing value")
plt.pyplot.show()


# For simplicity we will remove the variables with more than 60% of the values missing.\
# *We might re-use some of these variables if our model is not good enough.
# 

# In[69]:


train.drop(columns=["campaign_disc_ele", "date_first_activ",
 "forecast_base_bill_ele","forecast_base_bill_year",
"forecast_bill_12m", "forecast_cons"], inplace=True)


# Notice how the columns that we removed do not appear in the dataframe anymore.\
# **Note**: Showing the columns as a separate dataframe to facilitate visualization
# 

# In[ ]:


pd.DataFrame({"Dataframe columns": train.columns})


# #### Duplicates

# We want to make sure all the data we have is unique and we don't have any duplicated rows. For that, we're going to use the .duplicated()
# function in pandas.\
# This will tell us if there are any duplicated rows.
# 

# In[60]:


train[train.duplicated()]


# ### 5. Formatting data
# 

# #### Missing dates
# 
# There could be several ways in which we could deal with the missing dates.
# 
# One way, we could "engineer" the dates from known values. For example, the date_renewal is usually the same date as the date_modif_prod
# but one year ahead.
# 
# 
# The simplest way, we will replace the missing values with the median (the most frequent date). For numerical values, the built-in function .median()\
# can be used, but this will not work for dates or strings, so we will use a workaround using .value counts()
# 
# 

# In[73]:


train.loc[train["date_modif_prod"].isnull(),"date_modif_prod"] = train["date_modif_prod"].value_counts().index[0]
train.loc[train["date_end"].isnull(),"date_end"] = train["date_end"].value_counts().index[0]
train.loc[train["date_renewal"].isnull(),"date_renewal"] = train["date_renewal"].value_counts().index[0]


# #### Missing data
# 
# 
# We might have some prices missing for some companies and months
# 

# In[61]:


missing_data_precentage=(history_data.isnull().sum()/len(history_data.index)*100).plot(kind="bar",figsize=(18,10)) 
plt.pyplot.xlabel("variables") 
plt.pyplot.ylabel("precentage") 
plt.pyplot.show()


# There is not much data missing. Instead of removing the entries that are empty we will simply substitute them with the median .
# 
# 
# **Note**: We could use something slightly more complicated such as using the mean of the previous and following months to calculate the value of the
# missing month since the data does not vary much.

# In[62]:


history_data[history_data.isnull().any(axis=1)]


# In[63]:


history_data.loc[history_data["price_p1_var"].isnull(),"price_p1_var"]=history_data["price_p1_var"].median()
history_data.loc[history_data["price_p2_var"].isnull(),"price_p2_var"]=history_data["price_p2_var"].median()
history_data.loc[history_data["price_p3_var"].isnull(),"price_p3_var"]=history_data["price_p3_var"].median()
history_data.loc[history_data["price_p1_fix"].isnull(),"price_p1_fix"]=history_data["price_p1_fix"].median()
history_data.loc[history_data["price_p2_fix"].isnull(),"price_p2_fix"]=history_data["price_p2_fix"].median()
history_data.loc[history_data["price_p3_fix"].isnull(),"price_p3_fix"]=history_data["price_p3_fix"].median()


# In[64]:


train["date_activ"] = pd.to_datetime(train["date_activ"], format='%Y-%m-%d')
train["date_end"] = pd.to_datetime(train["date_end"], format='%Y-%m-%d')
train["date_modif_prod"] = pd.to_datetime(train["date_modif_prod"], format='%Y-%m-%d')
train["date_renewal"] = pd.to_datetime(train["date_renewal"], format='%Y-%m-%d')
history_data["price_date"]=pd.to_datetime(history_data["price_date"],format='%Y-%m-%d')


# In[74]:


history_data["price_date"] = pd.to_datetime(history_data["price_date"], format='%Y-%m-%d')


# In[77]:


fig, axs = plt.pyplot.subplots(nrows=7, figsize=(18,50))
# Plot boxplots
sns.boxplot((train["cons_12m"].dropna()), ax=axs[0])
sns.boxplot((train[train["has_gas"]==1]["cons_gas_12m"].dropna()), ax=axs[1])
sns.boxplot((train["cons_last_month"].dropna()), ax=axs[2])
sns.boxplot((train["forecast_cons_12m"].dropna()), ax=axs[3])
#sns.boxplot((train["forecast_cons_year"].dropna()), ax=axs[4])
sns.boxplot((train["forecast_meter_rent_12m"].dropna()), ax=axs[5])
sns.boxplot((train["imp_cons"].dropna()), ax=axs[6])
plt.pyplot.show()


# In[65]:


history_data.describe()


# We can see that there are negative values for price_p1_fix , price_p2_fix and price_p3_fix .
# 
# 
# Further exploring on those we can see there are only about 10 entries which are negative. This is more likely to be due to corrupted data rather than
# a "price discount".
# 
# We will replace the negative values with the median (most frequent value)
# 

# In[66]:


history_data[(history_data.price_p1_fix<0)|(history_data.price_p2_fix<0)|(history_data.price_p3_fix<0)]


# In[79]:


history_data.loc[history_data["price_p1_fix"] < 0,"price_p1_fix"] = history_data["price_p1_fix"].median()
history_data.loc[history_data["price_p2_fix"] < 0,"price_p2_fix"] = history_data["price_p2_fix"].median()
history_data.loc[history_data["price_p3_fix"] < 0,"price_p3_fix"] = history_data["price_p3_fix"].median()


# ### 8. Pickling
# 
# 
# Pickling is useful for applications where we need some degree of persistency in our data. Our program's state data can be saved to disk, so we can
# continue working on it later on.
# 
# 
# Make directory processed_data if it does not exist already
# 
# 

# In[82]:


cd "C:/Users/top/Desktop/BCGprojects"


# In[85]:


if not os.path.exists(os.path.join("C:/Users/top/Desktop/BCGprojects")):
    os.makedirs(os.path.join("C:/Users/top/Desktop/BCGprojects"))


# In[86]:


PICKLE_TRAIN_DIR = os.path.join("C:/Users/top/Desktop/BCGprojects", "train_data.pkl")
PICKLE_HISTORY_DIR = os.path.join("C:/Users/top/Desktop/BCGprojects", "history_data.pkl")


# In[87]:


pd.to_pickle(train, PICKLE_TRAIN_DIR)
pd.to_pickle(history_data, PICKLE_HISTORY_DIR)


# In[ ]:




