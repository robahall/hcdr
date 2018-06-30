
## Modules for feature cleaning Titanic dataset
import pandas as pd

def daysToYears(dfIn, dfOut):
    """Update education and one hot encode them"""

    years = dfIn['DAYS_BIRTH'] / -365
    dfOut = pd.concat([dfOut, years])
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
    dfOut = pd.DataFrame(appTrainDf['TARGET'])  #update this with numerical columns that don't need cleaning
    dfOut = daysToYears(dfIn, dfOut)
    dfOut = simplifyEducation(dfIn, dfOut)
    dfOut = simplifyFamily(dfIn, dfOut)
    dfOut = simplifyIncome(dfIn, dfOut)

    return dfOut