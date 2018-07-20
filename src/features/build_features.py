
## Modules for feature engineering hcdr dataset
from functools import reduce

import numpy as np
import pandas as pd


from scipy.stats  import skewtest

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder, scale
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer



## Base Selectors and Creators in scikit-learn Pipeline

class DFFeatureUnion(BaseEstimator, TransformerMixin):
    """Feature Union in Pipeline that is Pandas compatible."""

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for (name, t) in self.transformer_list:
            t.fit(X, y)
        return self

    def transform(self, X):
        Xts = [t.transform(X) for _, t in self.transformer_list]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
        return Xunion


class TypeSelector(BaseEstimator, TransformerMixin):
    """Selects if class is a boolean, a numeric, or a categorical feature. Pandas compatible"""

    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])

class ColumnExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xcols = X[self.cols]
        return Xcols



class DFsImputer(BaseEstimator, TransformerMixin):
    """Selects how missing data will be updated. Pandas compatible."""
    #TODO: need to create a dummy column to keep NaN features available.
    #TODO: Int() issue with simple imputer

    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.im = None
        self.statistics_ = None

    def fit(self, X, y=None):
        self.im = SimpleImputer(missing_values='NaN', strategy=self.strategy).fit(X)
        return self

    def transform(self, X):
        Xim = self.im.transform(X)
        Ximputed = pd.DataFrame(Xim, index=X.index, columns=X.columns)
        return Ximputed


#Clean Numeric

class DFLogTrans(BaseEstimator, TransformerMixin):
    """Tests skew of numeric dataset. If skew test p<=0.05, ensure values are log transformed."""
    #TODO: Doesn't always work as planned with skew test. Especially when distribution is close to uniform.
    #TODO: Test with different distributions

    def __init__(self, pvalue=[]):
        self.pvalue = pvalue

    def transform(self, X):
        if len(self.pvalue[0]) > 1:
            combine = zip(X, self.pvalue)
            Xtl = [np.log1p(np.absolute(X[XSeries])) if pvalues[0] <= 0.05 else X[XSeries] for XSeries, pvalues in combine]
            df =  reduce(lambda X1, X2: pd.concat((X1, X2), axis=1), Xtl)
            return df
        elif self.pvalue[0][0] <= 0.05:
            df = pd.DataFrame(np.log1p(np.absolute(X)))
            return df
        else:
            return X

    def fit(self, X, y=None):
        _, pvalue = skewtest(X)
        self.pvalue.append(pvalue)
        return self


class DFStandardScaler(BaseEstimator, TransformerMixin):
    """Standardizes distribution. Pandas compatible. """

    def __init__(self):
        self.ss = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.ss = StandardScaler()
        self.ss.fit(X)
        self.mean_ = pd.Series(self.ss.mean_, index = X.columns)
        self.scale_ = pd.Series(self.ss.scale_, index = X.columns)
        return self

    def transform(self, X):
        Xss = self.ss.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
        return Xscaled



#Clean Categorical

class CleanString(BaseEstimator, TransformerMixin):
    """Cleans String. Pandas compatible"""

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.apply(lambda s: s.str.replace(' ', ''))


class GetDummies(BaseEstimator, TransformerMixin):
    """Uses pandas get dummies over OneHotEncoder. Pandas compatible."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return pd.get_dummies(X)

#Clean Binary

class DFBinaryEncoder(BaseEstimator, TransformerMixin):
    """Converts boolean to 1 or 0"""

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return self





# Old code. Needs to be removed.




def standardizedIncome(dfIn, dfOut):
    """Log transform income and standardize"""
    nIncome = pd.Series(scale(np.log(dfIn['AMT_INCOME_TOTAL'])), name = 'scaledLogINC')
    dfOut = pd.concat([dfOut, nIncome], axis = 1)
    return dfOut


def engineerDays(dfIn, dfOut):
    """DAYS_EMPLOYEED is from when employment starts. Data is positively skewed.
    Need to log transform. Added flag columns for the anomly in data and people who have no job (only 2 in train set) .
    """

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
                     'NAME_CONTRACT_TYPE"', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'EDU_Academic_degree', 'EDU_Higher_education',
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
        dfOut = standardizedIncome(dfIn, dfOut)
        dfOut = engineerDays(dfIn, dfOut)
        dfOut = createEncoders(dfIn, dfOut)
        dfOut = simplifyEducation(dfIn, dfOut)
        dfOut = simplifyFamily(dfIn, dfOut)
        dfOut = simplifyIncome(dfIn, dfOut)
        dfOut = addExtSources(dfIn, dfOut)
        dfOut = cleanNames(dfOut)
        dfOut = createPolyFeatures(dfOut)
    else:
        dfOut = dfIn['SK_ID_CURR'] ## tags from test set
        dfOut = standardizedIncome(dfIn, dfOut)
        dfOut = engineerDays(dfIn, dfOut)
        dfOut = createEncoders(dfIn, dfOut)
        dfOut = simplifyEducation(dfIn, dfOut)
        dfOut = simplifyFamily(dfIn, dfOut)
        dfOut = simplifyIncome(dfIn, dfOut)
        dfOut = addExtSources(dfIn, dfOut)
        dfOut = dfOut.drop('CODE_GENDER', axis = 1) ## Need to fix this
        #print(dfOut.columns)
        dfOut = cleanNamesTest(dfOut)
        dfOut = createPolyFeatures(dfOut)

    return dfOut




