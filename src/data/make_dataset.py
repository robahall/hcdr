# -*- coding: utf-8 -*-

# Python functionality
import sys
#import click
import logging
#from pathlib import Path
#from dotenv import find_dotenv, load_dotenv

# Vectorization
import numpy as np
import pandas as pd

# Sci Kit learn packages for feature transforming
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
sys.path.append('../features/')
import build_features as bf

from sklearn.model_selection import train_test_split

#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())



##
## Initial Split functions
##

def loadSplit(input_filepath, testSize = 0.25):
    ## Split and save data

    initialDF = pd.read_csv(input_filepath)
    id = initialDF.iloc[:,0] # save id if need be for additional datasets
    X = initialDF.iloc[:,2:] # main data
    y = initialDF.iloc[:,1] # target data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize)

    return X_train, X_test, y_train, y_test

##
## Save data set
##

def saveTrainSet(X_train, y_train, output_filepath):

    pd.concat([X_train, y_train], axis = 1).to_csv(output_filepath)
    print("File successfully saved to: {}".format(output_filepath))





##
## Specific Feature Pre-processing
##

def ppDaysBirth(df):
    df[['DAYS_BIRTH']] = df[['DAYS_BIRTH']].apply(lambda x: np.absolute(x) / 365)  # convert from days since to years old
    return df

def ppEmployed(df):
    """DAYS_EMPLOYEED is from when employment starts. Data is positively skewed."""

    df['DAYS_EMPLOYED_ANOM'] = df["DAYS_EMPLOYED"] == 365243
    df['DAYS_EMPLOYED_ZERO'] = df["DAYS_EMPLOYED"] == 0
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({0: np.nan, np.log(365243): np.nan})
    return df

def ppCreditIncome(df):
    """Create feature of Credit to Income percentage"""
    df['creditIncomePct'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['creditIncomePct'] = np.log(df['creditIncomePct'].fillna(df['creditIncomePct'].median()))
    return df

def ppAnnuityIncome(df):
    """Amount of annualized payment to income percentage"""
    df['annuityIncomePct'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['annuityIncomePct'] = np.log(df.annuityIncomePct.fillna(df.annuityIncomePct.median()))
    return df

def ppCreditTerm(df):
    """Fraction of annualized payment to credit"""
    df['creditTerm'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['creditTerm'] = df['creditTerm'].fillna(df['creditTerm'].median())
    return df

def ppDaysEmployeed(df):
    """Ratio of days employed to days since they were born"""
    df['daysEmployedPct'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    return df

##
## Pipeline Functions
##

def fbNumericNonLog():
    """Scale without log transform"""
    featureNumeric = Pipeline([
        ('selector', bf.TypeSelector(np.number)),
        ('fillNaN', bf.DFImputer()),
        ('scaler', bf.DFStandardScaler())
    ])
    return featureNumeric


def fbNumericLog():
    """Scale with log transform"""
    featureNumeric = Pipeline([
        ('selector', bf.TypeSelector(np.number)),
        ('fillNaN', bf.DFImputer()),
        ('logTransform', bf.DFLogTrans()),
        ('scaler', bf.DFStandardScaler())
    ])
    return featureNumeric

def fbBinary():
    featureBinary = Pipeline([
        ('selector', bf.TypeSelector(np.bool)),
        ()

    ])
    return featureBinary


def fbCategorical():
    """Clean columns and convert to dummies"""

    featureCat = Pipeline([
        ('selector', bf.TypeSelector('object')),
        ('concat', bf.CleanString()),
        ('encoder', bf.GetDummies())
    ])
    return featureCat

##
## Full Pipeline of transformations
##

def fullPipe(df):

    """

    :param df:
    :return: df

    Current columns in [['DAYS_BIRTH', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED',
     #  'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE',
     #  'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_EMPLOYED.1', 'EXT_SOURCE_1',
     # 'EXT_SOURCE_2', 'EXT_SOURCE_3']]

     Current columns out [[
    """

    df = ppDaysBirth(df) # function to pre-process day of birth
    df = ppEmployed(df) # function to pre-process days
    df = ppCreditIncome(df) # function to create credit income column
    df = ppAnnuityIncome(df) # function to create annuity income column



    return df


##
## Run Main
##

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    X_train, X_test, y_train, y_test = loadSplit(input_filepath)

    print(fullPipe(X_train))

    saveTrainSet(X_train, y_train, output_filepath)


    #print(X_train.columns)
    #print(X_train)







if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automatically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())


    input = "../../data/interim/application_train.csv"
    output = "../../data/interim/application_train_pipeline.csv"
    main(input, output)
