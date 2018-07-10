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
sys.path.append('../features')
import build_features as bf



#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    featureExtraction = Pipeline([
        ## Should moved this to another function

        ('features', FeatureUnion(n_jobs=1, transformer_list=[

            # Split out numerics for processing
            ('numeric', Pipeline ([
                ('selector', bf.TypeSelector(np.number)),
                ('scaler', StandardScaler())
            ])),

            # Split out booleans for processing
            ('boolean', Pipeline([
                ('selector', bf.TypeSelector('bool'))
            ])),

            # Split out categorical
            ('categorical', Pipeline([
                ('objCat', bf.TransFormObjCat()),
                ('selector', bf.TypeSelector('category')),
                ('labeler', bf.StringIndexer()),
                ('encoder', OneHotEncoder(handle_unknown = 'ignore'))

            ]))


        ]))
    ])
    train = pd.read_csv(input_filepath)
    y = train.iloc[:, 0:2]
    X = train.iloc[:, 2:]

    featureExtraction.fit(X)

    y.concat

    return




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automatically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())


    input = "../data/interm/application_train.csv"
    output = "../data/interim/application_train_pipeline.csv"
    main(input, output)
