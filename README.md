{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "<img src=https://static.seattletimes.com/wp-content/uploads/2019/02/02062019_housing_160430-780x471.jpg width=\"1100\" height=\"400\">\n",
    "\n",
    "# A Regression Model that Appraise Homes remotely for ACME Bank\n",
    "\n",
    "**Authors**: Nate Walter, Douglas Lu, Shane Mangold. Arthur Kim\n",
    "\n",
    "* [Overview](#Overview)\n",
    "\n",
    "\n",
    "* [Business Problem](#Business-Problem)\n",
    "\n",
    "\n",
    "* [The Data](#The-Data)\n",
    "\n",
    "\n",
    "* [Methods](#methods)\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "## Overview\n",
    "This project predicts house prices for King County, Washington based on an existing dataset using \n",
    "regression modeling. Our findings will help ACME bank make home appraisles wihtout the need for a \n",
    "bank employee to enter the domicile. ACME can then use the predictions\n",
    "to set sales prices for homes to be put on the market at competitive market values based on the\n",
    "regression model's predictions.\n",
    "\n",
    "<img src=https://user-images.githubusercontent.com/66656063/125619329-48319b12-7456-46a7-b4a4-e27a6babbc6f.png width=\"500\" height=\"300\">\n",
    "\n",
    "http://seattlemag.com/sites/default/files/field/image/iStock-471370245.jpg \n",
    "## Business Problem\n",
    "A recent wave of COVID-related foreclosures has ACME Bank choose an alternative method to having an appraiser \n",
    "physically enter homes for inspection, so as to limit liability of employees being exposed to COVID-19. \n",
    "Instead, they have come to Group One Inc. for an accurate regression model predicting sales prices for each of their properties. \n",
    "\n",
    "\n",
    "## The Data\n",
    "The Dataset used is from King County, Washington between May 2014 and May 2015. It includes housing sales prices \n",
    "along with other descriptive information invovling the properties. \n",
    "\n",
    "* https://www.kaggle.com/harlfoxem/housesalesprediction\n",
    "\n",
    "\n",
    "## Methods\n",
    "This project uses  multiple linear regression in combination with feature engineering, recurssive feture illimination, \n",
    "and dummie regression to predict an unkown house's sales price, all while adhering to the assumptions of linear regression. \n",
    "Test are performed to discover any linear relationship between the dependent and independent variables. Multicollinearity is \n",
    "examined between the inependent variables. The normal distribution of errors and homeoscedasticity are also omni-present goals to be met.  \n",
    "Our model uses train-test split which allows us to evaluate whether it has the right balance of bias and variance. \n",
    "We use data visualization via Matplotlib and Seaborn taking advantage of histograms, heatmaaps and scatter plots to \n",
    "help in the exploritory data analysis process as well as evaluation and presentation. The three questions explored are:\n",
    "\n",
    "**1)** What features must be dropped to make an accurate predictive model?\n",
    "\n",
    "**2)** How accurate is our model to the true price of a King County home?\n",
    "\n",
    "**3)** Can our predictions substitute for in person appraisal?\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "/Users/nathanwalter/courses/phase2/phase2-project/the-real-phase2-project\n",
    "\n",
    "\n",
    " then run \"git rebase --continue\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (learn-env)",
   "language": "python",
   "name": "learn-env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
