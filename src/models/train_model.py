import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="ticks", color_codes=True)
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc


def loadSplit(input_filepath, testSize = 0.25):
    ## Split and save data
    ##TODO: Split after feature building.
    ##TODO: Concat test set to bottom then split back off.

    initialDF = pd.read_csv(input_filepath)
    id = initialDF.iloc[:,0] # save id if need be for additional datasets
    X = initialDF.iloc[:,2:] # main data
    y = initialDF.iloc[:,1] # target data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize)

    return X_train, X_test, y_train, y_test


## Build model
def trainModel(trainIn, target, testId, testIn, catIndices, nFolds = 5):
     # Extract feature names
    featureNames = list(trainIn.columns)

    print("Number of training columns is: " + str(len(featureNames)))

    # Convert to np arrays
    trainIn = np.array(trainIn)
    testFeatures = np.array(testIn)
    print("Number of training rows is: " + str(len(trainIn)))
    print("Number of test rows is: " + str(len(testIn)))

    # Initalize the kfold object
    kFolds = KFold(n_splits = nFolds, shuffle = True, random_state = 50)

    # Empty array for feature importances
    featureImportanceValues = np.zeros(len(featureNames))

    # Empty array for test predictions
    testPredictions = np.zeros(testFeatures.shape[0])

    # Empty array for out of fold validation predictions
    outOfFold = np.zeros(trainIn.shape[0])

    # Lists for recording validation and training scores
    validScores = []
    trainScores = []

    # Iterate through each fold
    for trainIndices, validIndices in kFolds.split(trainIn):

        # Training data for the fold
        trainFeatures, trainLabels = trainIn[trainIndices], target[trainIndices]
        # Validation data for the fold
        validFeatures, validLabels = trainIn[validIndices], target[validIndices]

        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary',
                                   class_weight = 'balanced', learning_rate = 0.05,
                                   reg_alpha = 0.1, reg_lambda = 0.1,
                                   subsample = 0.8, n_jobs = -1, random_state = 50)

        # Train the model
        model.fit(trainFeatures, trainLabels, eval_metric = 'auc',
                  eval_set = [(validFeatures, validLabels), (trainFeatures, trainLabels)],
                  eval_names = ['valid', 'train'], categorical_feature = catIndices,
                  early_stopping_rounds = 100, verbose = 200)

        # Record the best iteration
        bestIteration = model.best_iteration_

        # Record the feature importances
        featureImportanceValues += model.feature_importances_ / kFolds.n_splits

        # Make predictions
        testPredictions += model.predict_proba(testFeatures, num_iteration = bestIteration)[:, 1] / kFolds.n_splits
        # Record the out of fold predictions
        outOfFold[validIndices] = model.predict_proba(validFeatures, num_iteration = bestIteration)[:, 1]

        # Record the best score
        validScore = model.best_score_['valid']['auc']
        trainScore = model.best_score_['train']['auc']

        validScores.append(validScore)
        trainScores.append(trainScore)

        # Clean up memory
        gc.enable()
        del model, trainFeatures, validFeatures
        gc.collect()




    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': testId, 'TARGET': testPredictions})## Remove

    # Make the feature importance dataframe
    featureImportances = pd.DataFrame({'feature': featureNames, 'importance': featureImportanceValues})

    # Overall validation score
    validAuc = roc_auc_score(target, outOfFold)

    # Add the overall scores to the metrics
    validScores.append(validAuc)
    trainScores.append(np.mean(trainScores))

    # Needed for creating dataframe of validation scores
    foldNames = list(range(nFolds))
    foldNames.append('overall')

    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': foldNames,
                            'train': trainScores,
                            'valid': validScores})

    return submission, featureImportances, metrics
