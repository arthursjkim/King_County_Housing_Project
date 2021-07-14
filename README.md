<img src=https://static.seattletimes.com/wp-content/uploads/2019/02/02062019_housing_160430-780x471.jpg width="1100" height="400">

# A Regression Model that Appraise Homes remotely for ACME Bank

**Authors**: Nate Walter, Douglas Lu, Shane Mangold. Arthur Kim

* [Overview](#Overview)


* [Business Problem](#Business-Problem)


* [The Data](#The-Data)


* [Methods](#methods)




## Overview
This project predicts house prices for King County, Washington based on an existing dataset using 
regression modeling. Our findings will help ACME bank make home appraisles wihtout the need for a 
bank employee to enter the domicile. ACME can then use the predictions
to set sales prices for homes to be put on the market at competitive market values based on the
regression model's predictions.

<img src=https://user-images.githubusercontent.com/66656063/125619329-48319b12-7456-46a7-b4a4-e27a6babbc6f.png width="500" height="300">

http://seattlemag.com/sites/default/files/field/image/iStock-471370245.jpg 
## Business Problem
A recent wave of COVID-related foreclosures has ACME Bank choose an alternative method to having an appraiser 
physically enter homes for inspection, so as to limit liability of employees being exposed to COVID-19. 
Instead, they have come to Group One Inc. for an accurate regression model predicting sales prices for each of their properties. 


## The Data
The Dataset used is from King County, Washington between May 2014 and May 2015. It includes housing sales prices 
along with other descriptive information invovling the properties. 

* https://www.kaggle.com/harlfoxem/housesalesprediction


## Methods
This project uses  multiple linear regression in combination with feature engineering, recurssive feture illimination, 
and dummie regression to predict an unkown house's sales price, all while adhering to the assumptions of linear regression. 
Test are performed to discover any linear relationship between the dependent and independent variables. Multicollinearity is 
examined between the inependent variables. The normal distribution of errors and homeoscedasticity are also omni-present goals to be met.  
Our model uses train-test split which allows us to evaluate whether it has the right balance of bias and variance. 
We use data visualization via Matplotlib and Seaborn taking advantage of histograms, heatmaaps and scatter plots to 
help in the exploritory data analysis process as well as evaluation and presentation. The three questions explored are:

**1)** What features must be dropped to make an accurate predictive model?

**2)** How accurate is our model to the true price of a King County home?

**3)** Can our predictions substitute for in person appraisal?


