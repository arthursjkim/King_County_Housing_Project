#!/usr/bin/env python
# coding: utf-8

# # Phase 2 Group 1 Project

# # Import

# In[1]:


# import appropriate libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyRegressor
pd.set_option('display.max_columns', None)
get_ipython().run_line_magic('matplotlib', 'inline')


# # Review the Data File

# In[2]:


# read in the file
df = pd.read_csv('data/kc_house_data.csv')


# In[3]:


# check the first 5 entries in the data
df.head()


# In[4]:


# check the columns and nulls
df.info()


# ## Cleaning the Data

# In[5]:


# for year renovated, convert any houses that have been renovated to '1' to indicate true
# for any nulls, assume no renovation
df['yr_renovated'].fillna(0, inplace=True)
df['yr_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else x)


# In[6]:


df.rename(columns={'yr_renovated': 'if_renovated'}, inplace=True)


# In[7]:


# for any nulls, assume no waterfront
df['waterfront'].fillna(0, inplace=True)


# In[8]:


# for any nulls, assume no one viewed the property
df['view'].fillna(0, inplace=True)


# In[9]:


# clean up sqft_basement and convert to int
df['sqft_basement'] = df['sqft_basement'].replace({'?': np.nan}).astype(float)
df['sqft_basement'].fillna(df['sqft_living']-df['sqft_above'], inplace=True)


# In[10]:


# retrieve the months and year
df['month_of_date'] = pd.DatetimeIndex(df['date']).month
df['year_of_date'] = pd.DatetimeIndex(df['date']).year


# In[11]:


# convert yr_built to age of house by subtracting year the property was sold by the year it was built
# to create a more sensible column
df['age_of_house'] = df['year_of_date'] - df['yr_built']

# drop year of date because years are only 2014 and 2015, and will not impact our predicative model
# drop yr_built b/c it is redundant with age_of_house
df.drop(columns=['year_of_date'], inplace=True)
df.drop(columns=['yr_built'], inplace=True)


# In[12]:


# drop duplicates if any
df.drop_duplicates(inplace=True)


# In[13]:


# drop id and date columns
df.drop(columns=['id'], inplace=True)
df.drop(columns=['date'], inplace=True)


# In[14]:


# reset index
df.reset_index(inplace=True, drop=True)


# In[15]:


# convert some of the categorical data from floats to ints
df['waterfront'] = df['waterfront'].astype(int)
df['view'] = df['view'].astype(int)
df['sqft_basement'] = df['sqft_basement'].astype(int)
df['if_renovated'] = df['if_renovated'].astype(int)


# In[16]:


# check cleaned data
df.info()


# In[17]:


df.head()


# # EDA

# ### Target: Price

# In[18]:


# Since price is our target, we will explore first
# view distribution of price using histogram
df.hist(column=['price'], bins='auto')


# In[19]:


df.boxplot(column=['price'])


# In[20]:


# Because the data is skewed to the right, transform the price data using log
df['ln_price'] = np.log(df['price'])


# In[21]:


# view distribution of log base e for price using histogram
df.hist(column=['ln_price'], bins='auto')


# In[22]:


df.boxplot(column=['ln_price'])


# ### Predictors: Everything Else

# In[23]:


# sns.pairplot(df)


# In[24]:


df.columns


# In[25]:


# based on the pairplot, we can see which data are categorical and which are numeric
numeric = ['bedrooms',
           'bathrooms',
           'sqft_living',
           'sqft_lot',
           'sqft_above',
           'sqft_basement',
           'lat',
           'long',
           'sqft_living15',
           'sqft_lot15']

categorical = ['floors',
               'waterfront',
               'view',
               'condition',
               'grade',
               'if_renovated',
               'zipcode',
               'month_of_date']


# In[26]:


# Create a df with the target as the first column,
# then compute the correlation matrix
X = df.drop(['price', 'ln_price'], axis=1)
y = df['price']
ln_y = df['price']
heatmap_data = pd.concat([y, X], axis=1)
corr = heatmap_data.corr()

# Set up figure and axes
fig, ax = plt.subplots(figsize=(15, 15))

# Plot a heatmap of the correlation matrix, with both
# numbers and colors indicating the correlations
sns.heatmap(
    # Specifies the data to be plotted
    data=corr,
    # The mask means we only show half the values,
    # instead of showing duplicates. It's optional.
    mask=np.triu(np.ones_like(corr, dtype=bool)),
    # Specifies that we should use the existing axes
    ax=ax,
    # Specifies that we want labels, not just colors
    annot=True,
    # Customizes colorbar appearance
    cbar_kws={"label": "Correlation",
              "orientation": "horizontal", "pad": .2, "extend": "both"}
)

# Customize the plot appearance
ax.set_title("Heatmap of Correlation Between Attributes and Price")


# In[27]:


# reporting the correlation between price (target) and predictors
df.corr()['price'].drop(['ln_price']).map(abs).sort_values(ascending=False)


# In[28]:


# Create a df with the target as the first column,
# then compute the correlation matrix
X = df.drop(['price', 'ln_price'], axis=1)
ln_y = df['price']
heatmap_data = pd.concat([ln_y, X], axis=1)
corr = heatmap_data.corr()

# Set up figure and axes
fig, ax = plt.subplots(figsize=(15, 15))

# Plot a heatmap of the correlation matrix, with both
# numbers and colors indicating the correlations
sns.heatmap(
    # Specifies the data to be plotted
    data=corr,
    # The mask means we only show half the values,
    # instead of showing duplicates. It's optional.
    mask=np.triu(np.ones_like(corr, dtype=bool)),
    # Specifies that we should use the existing axes
    ax=ax,
    # Specifies that we want labels, not just colors
    annot=True,
    # Customizes colorbar appearance
    cbar_kws={"label": "Correlation",
              "orientation": "horizontal", "pad": .2, "extend": "both"}
)

# Customize the plot appearance
ax.set_title("Heatmap of Correlation Between Attributes and Price")


# In[29]:


# reporting the correlation between ln price (target) and predictors
df.corr()['ln_price'].drop(['price']).map(abs).sort_values(ascending=False)


# # Baseline Model

# ## Baseline: Data Manipulation

# In[30]:


# create a new df based on the cleaned df
bdf = df

# dummying categorical
bdf = pd.get_dummies(bdf, prefix=categorical, prefix_sep='_',
                     columns=categorical, drop_first=True)

# remove any periods from column names after dummying the data
modified_cols = []
for column in bdf.columns:
    modified_cols.append(column.replace(".", "_"))
bdf.columns = modified_cols


# ## Baseline: Training and Testing

# In[31]:


# create the appropriate x and y data sets
X = bdf.drop(['price', 'ln_price'], axis=1)
y = bdf['price']
ln_y = bdf['ln_price']

# standardizing X values
ss = StandardScaler()
ss.fit(X)
X_scaled = pd.DataFrame(ss.transform(X))
X_scaled.columns = X.columns

# create the training and testing samples for both the price and ln_price data
X_train1, X_test1, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42)
X_train2, X_test2, ln_y_train, ln_y_test = train_test_split(
    X_scaled, ln_y, test_size=0.20, random_state=42)


# In[32]:


# instatiate a splitter with n_splits = 10
splitter = ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)


# In[33]:


# create a dummy regressor model based on the target as price
baseline_model = DummyRegressor(strategy='mean')
baseline_model.fit(X_train1, y_train)

# setting up cross validation for price in a different way (x3)
baseline_scores = cross_validate(
    estimator=baseline_model,
    X=X_train1,
    y=y_train,
    return_train_score=True,
    cv=splitter
)

print("Train score:     ", np.median(baseline_scores["train_score"]))
print("Validation score:", np.median(baseline_scores["test_score"]))


# In[34]:


# create a dummy regressor model based on the target as price
baseline_ln_model = DummyRegressor(strategy='mean')
baseline_ln_model.fit(X_train2, ln_y_train)

# setting up cross validation for ln price in a different way (x3)
baseline_ln_scores = cross_validate(
    estimator=baseline_ln_model,
    X=X_train2,
    y=ln_y_train,
    return_train_score=True,
    cv=splitter
)

print("Train score:     ", np.median(baseline_ln_scores["train_score"]))
print("Validation score:", np.median(baseline_ln_scores["test_score"]))


# In[35]:


# find the test score using the dummy regression model
baseline_train_score = baseline_model.score(X_train1, y_train)
baseline_ln_train_score = baseline_ln_model.score(X_train2, ln_y_train)
baseline_test_score = baseline_model.score(X_test1, y_test)
baseline_ln_test_score = baseline_ln_model.score(X_test2, ln_y_test)

print("Baseline model train score:", baseline_train_score)
print("Baseline model test score:", baseline_test_score)
print()
print("Baseline model train ln score:", baseline_ln_train_score)
print("Baseline model test ln score:", baseline_ln_test_score)


# We find that the baseline model R-squared value is approximately 0.

# # Model 1

# Model 1 uses a multiple linear regression model of the data using all parameters, including the dummied out categorical parameters.

# ## Model 1: Data Manipulation

# In[36]:


# create a new df based on the cleaned df
m1df = df

# dummying categorical
m1df = pd.get_dummies(m1df, prefix=categorical,
                      prefix_sep='_', columns=categorical, drop_first=True)

# remove any periods from column names after dummying the data
modified_cols = []
for column in m1df.columns:
    modified_cols.append(column.replace(".", "_"))
m1df.columns = modified_cols


# ## Model 1: Training and Testing

# In[37]:


# create the appropriate x and y data sets
X = m1df.drop(['price', 'ln_price'], axis=1)
y = m1df['price']
ln_y = m1df['ln_price']

# standardizing X values
ss = StandardScaler()
ss.fit(X)
X_scaled = pd.DataFrame(ss.transform(X))
X_scaled.columns = X.columns

# create the training and testing samples for both the price and ln_price data
X_train1, X_test1, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42)
X_train2, X_test2, ln_y_train, ln_y_test = train_test_split(
    X_scaled, ln_y, test_size=0.20, random_state=42)


# In[38]:


# instatiate a splitter with n_splits = 10
splitter = ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)


# In[39]:


# create linear regression model for price and setting up cross validation
m1df_model = LinearRegression()
m1df_model.fit(X_train1, y_train)

# setting up cross validation for price in a different way
m1df_scores = cross_validate(
    estimator=m1df_model,
    X=X_train1,
    y=y_train,
    return_train_score=True,
    cv=splitter
)

print("Train score:     ", np.median(m1df_scores["train_score"]))
print("Validation score:", np.median(m1df_scores["test_score"]))


# In[40]:


# create linear regression model for ln price and setting up cross validation (x3)
m1df_ln_model = LinearRegression()
m1df_ln_model.fit(X_train2, ln_y_train)

# setting up cross validation for ln price in a different way (x3)
ln_scores = cross_validate(
    estimator=m1df_ln_model,
    X=X_train2,
    y=ln_y_train,
    return_train_score=True,
    cv=splitter
)

print("Train score:     ", np.median(ln_scores["train_score"]))
print("Validation score:", np.median(ln_scores["test_score"]))


# In[41]:


# find the test score using the linear model
m1df_train_score = m1df_model.score(X_train1, y_train)
m1df_ln_train_score = m1df_ln_model.score(X_train2, ln_y_train)
m1df_test_score = m1df_model.score(X_test1, y_test)
m1df_ln_test_score = m1df_ln_model.score(X_test2, ln_y_test)

print("m1df model train score:", m1df_train_score)
print("m1df model test score:", m1df_test_score)
print("score varied by", round(
    abs(m1df_test_score - m1df_train_score)/m1df_train_score*100), "%")
print()
print("m1df ln model train score:", m1df_ln_train_score)
print("m1df ln model test score:", m1df_ln_test_score)
print("score varied by", round(abs(m1df_ln_test_score -
      m1df_ln_train_score)/m1df_ln_train_score*100), "%")


# We see that Model 1 outputted a higher R-squared value than the Baseline model between 0.8 and 0.9. In addition, Model 1 train and test scores varied by < 5%. The model that used the log-transformed price data had a higher R-squared value than the untransformed price data model.

# ## Model 1: OLS

# In[42]:


# set target
target = 'price'
ln_target = 'ln_price'

# concate the X and y of the train tests to apply the OLS to the full train data
train_df = pd.concat([X_train1, y_train], axis=1)
train_ln_df = pd.concat([X_train2, ln_y_train], axis=1)

# join the column names with "+"
columns = "+".join(train_df.drop(['price'], axis=1).columns)
columns_ln = "+".join(train_ln_df.drop(['ln_price'], axis=1).columns)


# In[43]:


# set formulas
formula = target + '~' + columns
ln_formula = ln_target + '~' + columns


# In[44]:


# create the OLS
m1df_ols = ols(formula, train_df).fit()
m1df_ln_ols = ols(ln_formula, train_ln_df).fit()


# In[45]:


# report out the OLS
m1df_ols.summary()


# In[46]:


# report out the OLS
m1df_ln_ols.summary()


# In[47]:


# create a dataframe for the parameters and pvalues
results = pd.DataFrame(m1df_ln_ols.pvalues)
results.reset_index(inplace=True)


# In[48]:


# rename the columns
results.rename(columns={'index': 'parameter', 0: 'pvalue'}, inplace=True)


# In[49]:


# create a list of parameters that have a pvalue > 0.05
parameters = list(results[results['pvalue'] > 0.05]['parameter'])


# # Model 2

# Model 2 is similar to Model 1, except the parameters that had a pvalue > 0.05 from the Model 1 log-transformed analysis were removed in Model 2.

# ## Model 2: Data Manipulation

# In[50]:


# create a new df based on the cleaned df
m2df = df

# dummying categorical
m2df = pd.get_dummies(m2df, prefix=categorical,
                      prefix_sep='_', columns=categorical, drop_first=True)

# drop price data, since ln_price will be used
m2df = m2df.drop(['price'], axis=1)

# remove any periods from column names after dummying the data
modified_cols = []
for column in m2df.columns:
    modified_cols.append(column.replace(".", "_"))
m2df.columns = modified_cols

# drop columns that had a high p-value from Model 2 OLS
m2df = m2df.drop(parameters, axis=1)


# ## Model 2: Training and Testing

# In[51]:


# create the appropriate x and y data sets
X = m2df.drop(['ln_price'], axis=1)
y = m2df['ln_price']

# standardizing X values
ss = StandardScaler()
ss.fit(X)
X_scaled = pd.DataFrame(ss.transform(X))
X_scaled.columns = X.columns

# create the training and testing samples
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42)


# In[52]:


# instatiate a splitter with n_splits = 10
splitter = ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)


# In[53]:


# create linear regression model for price and setting up cross validation (x3)
m2df_model = LinearRegression()
m2df_model.fit(X_train, y_train)

# setting up cross validation for price in a different way (x3)
m2df_scores = cross_validate(
    estimator=m2df_model,
    X=X_train,
    y=y_train,
    return_train_score=True,
    cv=splitter
)

print("Train score:     ", np.median(m2df_scores["train_score"]))
print("Validation score:", np.median(m2df_scores["test_score"]))


# In[54]:


# find the test score using the linear model
m2df_train_score = m2df_model.score(X_train, y_train)
m2df_test_score = m2df_model.score(X_test, y_test)


print("m2df model train score:", m2df_train_score)
print("m2df model test score:", m2df_test_score)
print("score varied by", round(
    abs(m2df_test_score - m2df_train_score)/m2df_train_score*100), "%")


# In[55]:


# compare Model 1 and 2
print("Model 1:")
print("m1df ln model train score:", m1df_ln_train_score)
print("m1df ln model test score:", m1df_ln_test_score)
print()
print("Model 2:")
print("m2df model train score:", m2df_train_score)
print("m2df model test score:", m2df_test_score)


# xxx

# ## Model 2: OLS

# In[56]:


# set target
target = 'ln_price'

# concate the X and y of the train tests to apply the OLS to the full train data
train_df = pd.concat([X_train, y_train], axis=1)

# join the column names with "+"
columns = "+".join(train_df.drop(['ln_price'], axis=1).columns)


# In[57]:


# set formulas
formula = target + '~' + columns


# In[58]:


# create the OLS
m2df_ols = ols(formula, train_df).fit()


# In[59]:


# report out the OLS
m2df_ols.summary()


# # Model 3

# Model 3 uses the same dataframe structure as Model 1, but introduces recursive feature elimination (REF) of varying n_parameters to create the model. Since Model 1 showed that the log-transformed price model performed better than untransformed price model, we assumed that all models going forward should use the log-transformed price model.

# ## Model 3: Data Manipulation

# In[60]:


# create a new df based on the cleaned df
m3df = df

# drop price data, since ln_price will be used
m3df = m3df.drop(['price'], axis=1)

# dummying categorical
m3df = pd.get_dummies(m3df, prefix=categorical,
                      prefix_sep='_', columns=categorical, drop_first=True)

# remove any periods from column names after dummying the data
modified_cols = []
for column in m3df.columns:
    modified_cols.append(column.replace(".", "_"))
m3df.columns = modified_cols


# ## Model 3: Recursive Feature Elimination (REF)

# In[61]:


# determine how many columns the dataframe has
len(m3df.columns)


# In[62]:


# determine n for REF
n = [10, 20, 50, 100]

key_cols = {}

for x in n:
    # instatiate Linear Regression
    lr_rfe = LinearRegression()
    select = RFE(lr_rfe, n_features_to_select=x)

    # instatiate StandardScaler to standardize ln_price
    ss = StandardScaler()
    ss.fit(m3df.drop(['ln_price'], axis=1))
    m3df_scaled = ss.transform(m3df.drop(['ln_price'], axis=1))

    # fit model to RFE
    select.fit(X=m3df_scaled, y=m3df['ln_price'])

    # obtain the indexes where select.support_ is true
    true_indexes = [i for i, x in enumerate(select.support_) if x]

    # create a list of all column names matched with index and add to dictionary
    key_columns = []
    for i in true_indexes:
        key_columns.append(m3df.drop(['ln_price'], axis=1).columns[i])
    key_cols["key_cols_{0}".format(x)] = key_columns


# ## Model 3: Training and Testing, n = 10

# In[63]:


# create the appropriate x and y data sets
X = m3df[key_cols['key_cols_10']]
y = m3df['ln_price']

# standardizing X values
ss = StandardScaler()
ss.fit(X)
X_scaled = pd.DataFrame(ss.transform(X))
X_scaled.columns = X.columns

# create the training and testing samples
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42)


# In[64]:


# instatiate a splitter with n_splits = 10
splitter = ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)


# In[65]:


# create linear regression model for price and setting up cross validation (x3)
m3df_model = LinearRegression()
m3df_model.fit(X_train, y_train)

# setting up cross validation for price in a different way (x3)
m3df_scores = cross_validate(
    estimator=m3df_model,
    X=X_train,
    y=y_train,
    return_train_score=True,
    cv=splitter
)

print("Train score:     ", np.median(m3df_scores["train_score"]))
print("Validation score:", np.median(m3df_scores["test_score"]))


# In[66]:


# find the test score using the linear model
m3df_train_score_10 = m3df_model.score(X_train, y_train)
m3df_test_score_10 = m3df_model.score(X_test, y_test)


print("m3df model train score:", m3df_train_score_10)
print("m3df model test score:", m3df_test_score_10)
print("score varied by", round(abs(m3df_test_score_10 -
      m3df_train_score_10)/m3df_train_score_10*100), "%")


# ## Model 3: Training and Testing, n = 20

# In[67]:


# create the appropriate x and y data sets
X = m3df[key_cols['key_cols_20']]
y = m3df['ln_price']

# standardizing X values
ss = StandardScaler()
ss.fit(X)
X_scaled = pd.DataFrame(ss.transform(X))
X_scaled.columns = X.columns

# create the training and testing samples
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42)


# In[68]:


# instatiate a splitter with n_splits = 10
splitter = ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)


# In[69]:


# create linear regression model for price and setting up cross validation (x3)
m3df_model = LinearRegression()
m3df_model.fit(X_train, y_train)

# setting up cross validation for price in a different way (x3)
m3df_scores = cross_validate(
    estimator=m3df_model,
    X=X_train,
    y=y_train,
    return_train_score=True,
    cv=splitter
)

print("Train score:     ", np.median(m3df_scores["train_score"]))
print("Validation score:", np.median(m3df_scores["test_score"]))


# In[70]:


# find the test score using the linear model
m3df_train_score_20 = m3df_model.score(X_train, y_train)
m3df_test_score_20 = m3df_model.score(X_test, y_test)


print("m3df model train score:", m3df_train_score_20)
print("m3df model test score:", m3df_test_score_20)
print("score varied by", round(abs(m3df_test_score_20 -
      m3df_train_score_20)/m3df_train_score_20*100), "%")


# ## Model 3: Training and Testing, n = 50

# In[71]:


# create the appropriate x and y data sets
X = m3df[key_cols['key_cols_50']]
y = m3df['ln_price']

# standardizing X values
ss = StandardScaler()
ss.fit(X)
X_scaled = pd.DataFrame(ss.transform(X))
X_scaled.columns = X.columns

# create the training and testing samples
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42)


# In[72]:


# instatiate a splitter with n_splits = 10
splitter = ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)


# In[73]:


# create linear regression model for price and setting up cross validation (x3)
m3df_model = LinearRegression()
m3df_model.fit(X_train, y_train)

# setting up cross validation for price in a different way (x3)
m3df_scores = cross_validate(
    estimator=m3df_model,
    X=X_train,
    y=y_train,
    return_train_score=True,
    cv=splitter
)

print("Train score:     ", np.median(m3df_scores["train_score"]))
print("Validation score:", np.median(m3df_scores["test_score"]))


# In[74]:


# find the test score using the linear model
m3df_train_score_50 = m3df_model.score(X_train, y_train)
m3df_test_score_50 = m3df_model.score(X_test, y_test)


print("m3df model train score:", m3df_train_score_50)
print("m3df model test score:", m3df_test_score_50)
print("score varied by", round(abs(m3df_test_score_50 -
      m3df_train_score_50)/m3df_train_score_50*100), "%")


# ## Model 3: Training and Testing, n = 100

# In[75]:


# create the appropriate x and y data sets
X = m3df[key_cols['key_cols_100']]
y = m3df['ln_price']

# standardizing X values
ss = StandardScaler()
ss.fit(X)
X_scaled = pd.DataFrame(ss.transform(X))
X_scaled.columns = X.columns

# create the training and testing samples
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42)


# In[76]:


# instatiate a splitter with n_splits = 10
splitter = ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)


# In[77]:


# create linear regression model for price and setting up cross validation (x3)
m3df_model = LinearRegression()
m3df_model.fit(X_train, y_train)

# setting up cross validation for price in a different way (x3)
m3df_scores = cross_validate(
    estimator=m3df_model,
    X=X_train,
    y=y_train,
    return_train_score=True,
    cv=splitter
)

print("Train score:     ", np.median(m3df_scores["train_score"]))
print("Validation score:", np.median(m3df_scores["test_score"]))


# In[78]:


# find the test score using the linear model
m3df_train_score_100 = m3df_model.score(X_train, y_train)
m3df_test_score_100 = m3df_model.score(X_test, y_test)


print("m3df model train score:", m3df_train_score_100)
print("m3df model test score:", m3df_test_score_100)
print("score varied by", round(abs(m3df_test_score_100 -
      m3df_train_score_100)/m3df_train_score_100*100), "%")


# ## Model 3: Summary

# In[79]:


print("Model 3: Varying REF n:")
print()
print("When we include all columns (based on Model 1):")
print("m1df model train score:", m1df_ln_train_score)
print("m1df model test score:", m1df_ln_test_score)
print()
print("When n = 10:")
print("m3df model train score:", m3df_train_score_10)
print("m3df model test score:", m3df_test_score_10)
print()
print("When n = 10:")
print("m3df model train score:", m3df_train_score_20)
print("m3df model test score:", m3df_test_score_20)
print()
print("When n = 50:")
print("m3df model train score:", m3df_train_score_50)
print("m3df model test score:", m3df_test_score_50)
print()
print("When n = 100:")
print("m3df model train score:", m3df_train_score_100)
print("m3df model test score:", m3df_test_score_100)


# # Model 4 Binomial Feature Engineering

# ## Binomial Feature Engineering

# In[80]:


#X_train2, X_test2, ln_y_train, ln_y_test

pf = PolynomialFeatures(degree=2)
pf.fit(X_train2)


# In[81]:


pdf_train = pd.DataFrame(pf.transform(
    X_train2), columns=pf.get_feature_names())
pdf_test = pd.DataFrame(pf.transform(X_test2), columns=pf.get_feature_names())


# ### Pearson Correlation Coefficient Filtering

# In[82]:


correlations = pdf_train.corrwith(ln_y_train)
correlations_df = pd.DataFrame(correlations)
correlations_df.head()


# In[83]:


correlations_df.dropna(inplace=True)


# In[84]:


correlations_df[0].map(abs)
correlations_df.sort_values(by=0, ascending=False, inplace=True)


# In[85]:


features = list(correlations_df.index[:250]) + ['1']
print(features)


# ### Recursive Feature Elimination

# In[86]:


lr_rfe = LinearRegression()
select = RFE(lr_rfe, n_features_to_select=250)
select.fit(X=pdf_train[features], y=y_train)


# In[87]:


pdf_keepers = [x[0] for x in zip(pdf_train.columns, select.support_)
               if x[1] == True]


# In[88]:


len(pdf_keepers)


# In[89]:


lr_rfe.fit(X=pdf_train[pdf_keepers], y=y_train)
lr_rfe.score(pdf_train[pdf_keepers], y=y_train), lr_rfe.score(pdf_test[pdf_keepers], y=y_test), (lr_rfe.score(
    pdf_train[pdf_keepers], y=y_train)-lr_rfe.score(pdf_test[pdf_keepers], y=y_test))


# ### Cross Validation

# In[90]:


check = [5, 10, 20]
cross_results = {x:
                 (np.median(cross_val_score(lr_rfe, X, ln_y, cv=x)),
                  np.std(cross_val_score(lr_rfe, X, ln_y, cv=x)))
                 for x in check}


# In[91]:


cross_results
