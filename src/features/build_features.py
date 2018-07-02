
## Modules for feature engineering hcdr dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder

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

def createEncoders(dfIn, dfOut):
    '''Convert from object type to binary'''
    labelEncode = LabelEncoder()
    leCount = 0

    for col in dfIn:
        if dfIn[col].dtype == 'object':
            if len(list(dfIn[col].unique())) <= 2:
                labelEncode.fit(dfIn[col])
                dfOut[col] = labelEncode.transform(dfIn[col])

                leCount += 1
    print("{:d} columns were label encoded".format(leCount))
    return dfOut

def makeCreditIncome(dfIn, dfOut):
    '''Create feature of Credit to Income percentage'''
    creditIncome = dfIn['AMT_CREDIT'] / dfIn['AMT_INCOME_TOTAL']
    creditIncome = pd.Series(np.log(creditIncome.fillna(creditIncome.median())), name='creditIncomePct')
    dfOut = pd.concat([dfOut, creditIncome], axis=1)
    return dfOut

def makeAnnuityIncome(dfIn, dfOut):
    '''Amount of annualized payment to income percentage '''
    annuityIncomePct = dfIn['AMT_ANNUITY'] / dfIn['AMT_INCOME_TOTAL']
    annuityIncomePct = pd.Series(np.log(annuityIncomePct.fillna(annuityIncomePct.median())), name='annuityIncomePct')
    dfOut = pd.concat([dfOut, annuityIncomePct], axis=1)
    return dfOut

def makeCreditTerm(dfIn, dfOut):
    '''Fraction of annualized payment to credit'''
    dfOut['creditTerm'] = dfIn['AMT_ANNUITY'] / dfIn['AMT_CREDIT']
    dfOut['creditTerm'] = dfOut['creditTerm'].fillna(dfOut['creditTerm'].median())
    return dfOut

def makeDaysEmployeed(dfIn, dfOut):
    '''Ratio of days employed to days since they were born'''
    dfOut['daysEmployedPct'] = dfIn['DAYS_EMPLOYED'] / dfIn['DAYS_BIRTH']
    return dfOut

def addExtSources(dfIn, dfOut):
    '''Add External Sources'''
    dfOut['EXT_SOURCE_1'] = dfIn['EXT_SOURCE_1']
    dfOut['EXT_SOURCE_2'] = dfIn['EXT_SOURCE_2']
    dfOut['EXT_SOURCE_3'] = dfIn['EXT_SOURCE_3']
    return dfOut

def cleanNames(dfOut):
    dfOut.columns = ['TARGET', 'DAYS_BIRTH', 'scaledLogINC', 'DAYS_EMPLOYED_ANOM', 'DAYS_EMPLOYED_ZERO', 'DAYS_EMPLOYED',
                     'NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'EDU_Academic_degree', 'EDU_Higher_education',
                     'EDU_Incomplete_higher', 'EDU_Lower_secondary', 'EDU_Secondary_special', 'FAM_Civil_marriage',
                     'FAM_Married', 'FAM_Separated', 'FAM_Single', 'FAM_Unknown', 'FAM_Widow', 'INC_Businessman',
                     'INC_Commercial', 'INC_Maternity', 'INC_Pensioner', 'INC_State', 'INC_Student', 'INC_Unemployed',
                     'INC_Working', 'creditIncomePct', 'annuityIncomePct', 'creditTerm', 'daysEmployedPct', 'EXT_SOURCE_1',
                     'EXT_SOURCE_2', 'EXT_SOURCE_3']
    return dfOut

def cleanNamesTest(dfOut):
    dfOut.columns = ['SK_ID_CURR', 'DAYS_BIRTH', 'scaledLogINC', 'DAYS_EMPLOYED_ANOM', 'DAYS_EMPLOYED_ZERO', 'DAYS_EMPLOYED',
                     'NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'EDU_Academic_degree', 'EDU_Higher_education',
                     'EDU_Incomplete_higher', 'EDU_Lower_secondary', 'EDU_Secondary_special', 'FAM_Civil_marriage',
                     'FAM_Married', 'FAM_Separated', 'FAM_Single', 'FAM_Widow', 'INC_Businessman',
                     'INC_Commercial',  'INC_Pensioner', 'INC_State', 'INC_Student', 'INC_Unemployed',
                     'INC_Working', 'creditIncomePct', 'annuityIncomePct', 'creditTerm', 'daysEmployedPct', 'EXT_SOURCE_1',
                     'EXT_SOURCE_2', 'EXT_SOURCE_3']
    return dfOut

def createPolyFeatures(dfOut):
    '''These features have been created from ANOVA in 4.0'''

    dfOut['empAnomToNameContract'] = dfOut['DAYS_EMPLOYED_ANOM']*dfOut['NAME_CONTRACT_TYPE']
    dfOut['daysEmployedToCreditIncomePct'] = dfOut['DAYS_EMPLOYED']*dfOut['creditIncomePct']
    dfOut['daysEmployedToAnnuityIncomePct'] = dfOut['DAYS_EMPLOYED']*dfOut['annuityIncomePct']
    dfOut['daysEmployedToCreditTerm'] = dfOut['DAYS_EMPLOYED'] * dfOut['creditTerm']
    dfOut['daysEmployedToDaysEmployedPct'] = dfOut['DAYS_EMPLOYED'] * dfOut['daysEmployedPct']
    dfOut['daysEmployedAnomToEduLowerSecondary'] = dfOut['DAYS_EMPLOYED_ANOM'] * dfOut['EDU_Lower_secondary']
    dfOut['daysEmployedAnomToFamMarried'] = dfOut['DAYS_EMPLOYED_ANOM'] * dfOut['FAM_Married']
    dfOut['daysEmployedAnomToFamSingle'] = dfOut['DAYS_EMPLOYED_ANOM'] * dfOut['FAM_Single']
    dfOut['daysEmployedAnomToIncBusinessman'] = dfOut['DAYS_EMPLOYED_ANOM'] * dfOut['INC_Businessman']
    dfOut['daysEmployedAnomToIncCommercial'] = dfOut['DAYS_EMPLOYED_ANOM'] * dfOut['INC_Commercial']
    dfOut['daysEmployedAnomToCreditIncomePct'] = dfOut['DAYS_EMPLOYED_ANOM'] * dfOut['creditIncomePct']
    dfOut['daysEmployedAnomToAnnuityIncomePct'] = dfOut['DAYS_EMPLOYED_ANOM'] * dfOut['annuityIncomePct']
    dfOut['daysEmployedAnomToCreditTerm'] = dfOut['DAYS_EMPLOYED_ANOM'] * dfOut['creditTerm']
    dfOut['daysEmployedAnomToDaysEmployedPct'] = dfOut['DAYS_EMPLOYED_ANOM'] * dfOut['daysEmployedPct']
    dfOut['eduHigherEducationToDaysEmployed'] = dfOut['EDU_Higher_education'] * dfOut['DAYS_EMPLOYED']
    dfOut['eduHigherEducationToCreditIncomePct'] = dfOut['EDU_Higher_education'] * dfOut['creditIncomePct']
    dfOut['eduHigherEducationToAnnuityIncomePct'] = dfOut['EDU_Higher_education'] * dfOut['annuityIncomePct']
    dfOut['eduHigherEducationToCreditTerm'] = dfOut['EDU_Higher_education'] * dfOut['creditTerm']
    dfOut['eduHigherEducationToCreditIncomePct'] = dfOut['EDU_Higher_education'] * dfOut['creditIncomePct']
    dfOut['eduHigherEducationToCreditIncomePct'] = dfOut['EDU_Higher_education'] * dfOut['creditIncomePct']
    dfOut['eduHigherEducationToCreditIncomePct'] = dfOut['EDU_Higher_education'] * dfOut['creditIncomePct']
    return dfOut


def executeFeatures(dfIn, train = True):
    """One education, family, income."""

    if train == True:
        dfOut = dfIn['TARGET'] #update this with numerical columns that don't need cleaning
        dfOut = daysToYears(dfIn, dfOut)
        dfOut = standardizedIncome(dfIn, dfOut)
        dfOut = engineerDays(dfIn, dfOut)
        dfOut = createEncoders(dfIn, dfOut)
        dfOut = simplifyEducation(dfIn, dfOut)
        dfOut = simplifyFamily(dfIn, dfOut)
        dfOut = simplifyIncome(dfIn, dfOut)
        dfOut = makeCreditIncome(dfIn, dfOut)
        dfOut = makeAnnuityIncome(dfIn, dfOut)
        dfOut = makeCreditTerm(dfIn, dfOut)
        dfOut = makeDaysEmployeed(dfIn, dfOut)
        dfOut = addExtSources(dfIn, dfOut)
        dfOut = cleanNames(dfOut)
        dfOut = createPolyFeatures(dfOut)
    else:
        dfOut = dfIn['SK_ID_CURR'] ## tags from test set
        dfOut = daysToYears(dfIn, dfOut)
        dfOut = standardizedIncome(dfIn, dfOut)
        dfOut = engineerDays(dfIn, dfOut)
        dfOut = createEncoders(dfIn, dfOut)
        dfOut = simplifyEducation(dfIn, dfOut)
        dfOut = simplifyFamily(dfIn, dfOut)
        dfOut = simplifyIncome(dfIn, dfOut)
        dfOut = makeCreditIncome(dfIn, dfOut)
        dfOut = makeAnnuityIncome(dfIn, dfOut)
        dfOut = makeCreditTerm(dfIn, dfOut)
        dfOut = makeDaysEmployeed(dfIn, dfOut)
        dfOut = addExtSources(dfIn, dfOut)
        dfOut = dfOut.drop('CODE_GENDER', axis = 1) ## Need to fix this
        #print(dfOut.columns)
        dfOut = cleanNamesTest(dfOut)
        dfOut = createPolyFeatures(dfOut)

    return dfOut