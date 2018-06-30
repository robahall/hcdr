
## Modules for feature engineering hcdr dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import scale

def daysToYears(dfIn, dfOut):
    """Update education and one hot encode them"""

    years = dfIn['DAYS_BIRTH'] / -365
    dfOut = pd.concat([dfOut, years], axis = 1)
    return dfOut

def normalizeIncome(dfIn, dfOut):
    """OLD - Log transform income and standardize income"""
    nIncome = np.log(dfIn['AMT_INCOME_TOTAL'])
    nIncome.rename('logAMT_INCOME', inplace = True)
    nMean = dfIn['AMT_INCOME_TOTAL'].mean() ## Finds mean
    nStd = dfIn['AMT_INCOME_TOTAL'].std() ## Finds standard deviation
    nIncome = (nIncome - nMean)/nStd ## Standardization
    dfOut = pd.concat([dfOut, nIncome], axis = 1)
    return dfOut

def standardizedIncome(dfIn, dfOut):
    """Log transform income and standardize"""
    nIncome = pd.Series(scale(np.log(dfIn['AMT_INCOME_TOTAL'])), name = 'scaledLogINC')
    dfOut = pd.concat([dfOut, nIncome], axis = 1)
    return dfOut


def engineerDays(dfIn, dfOut):
    '''DAYS_EMPLOYEED is from when employment starts. Data is positively skewed.
    Need to log transform. Added flag columns for the anomly in data and people who have no job (only 2 in train set) .'''

    imp = Imputer(missing_values='NaN', strategy='mean', axis=1)

    dfOut['DAYS_EMPLOYED_ANOM'] = dfIn["DAYS_EMPLOYED"] == 365243
    dfOut['DAYS_EMPLOYED_ZERO'] = dfIn["DAYS_EMPLOYED"] == 0
    dfOut['DAYS_EMPLOYED'] = dfIn['DAYS_EMPLOYED'].replace({0: np.nan})
    dfOut['DAYS_EMPLOYED'] = (dfOut['DAYS_EMPLOYED'] * (-1)).apply(np.log)
    dfOut['DAYS_EMPLOYED'] = dfOut['DAYS_EMPLOYED'].replace({np.log(365243): np.nan})
    dfOut['DAYS_EMPLOYED'] = dfOut['DAYS_EMPLOYED'].fillna(dfOut['DAYS_EMPLOYED'].mean())
    dfOut['DAYS_EMPLOYED'] = scale(dfOut['DAYS_EMPLOYED'])

    return dfOut

def simplifyEducation(dfIn, dfOut):
    """Update education and one hot encode them"""

    edu = pd.get_dummies(dfIn.NAME_EDUCATION_TYPE, prefix = 'EDU')
    dfOut = pd.concat([dfOut, edu], axis = 1)
    return dfOut

def simplifyFamily(dfIn, dfOut):
    """Update Family and one hot encode them"""

    fam = pd.get_dummies(dfIn.NAME_FAMILY_STATUS, prefix = 'FAM')
    dfOut = pd.concat([dfOut, fam], axis = 1)
    return dfOut


def simplifyIncome(dfIn, dfOut):
    "Update Income and one hot encode them"

    inc = pd.get_dummies(dfIn.NAME_INCOME_TYPE, prefix='INC')
    dfOut = pd.concat([dfOut, inc], axis=1)
    return dfOut


def executeFeatures(dfIn):
    """One education, family, income."""

    dfOut = dfIn['TARGET']  #update this with numerical columns that don't need cleaning
    dfOut = daysToYears(dfIn, dfOut)
    #dfOut = normalizeIncome(dfIn, dfOut)
    dfOut = standardizedIncome(dfIn, dfOut)
    dfOut = engineerDays(dfIn, dfOut)
    dfOut = simplifyEducation(dfIn, dfOut)
    dfOut = simplifyFamily(dfIn, dfOut)
    dfOut = simplifyIncome(dfIn, dfOut)

    return dfOut