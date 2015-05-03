import argparse
import pandas as pd
import numpy as np
import random
import ast
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
import pdb
import math
from collections import defaultdict
from numpy.core.umath import sign
from sklearn.svm.classes import SVC
from sklearn.linear_model.coordinate_descent import Lasso
from sklearn import linear_model

from prepare_data import *
from analysis import *
from util import *

#Seed the random so that we can compare code changes
random.seed(12345)




if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--subsample", help="Percentage of the dataset to use", type=float, default=1, required=False)
    argparser.add_argument("--penalty", help="Penalty Function for L.R. (l1 or l2)", type=str, default="l2", required=False)
    argparser.add_argument("--cVal", help="C value for L.R.", type=float, default=10, required=False)
    argparser.add_argument("--twoStage", help="Splits the classification into two stages, position and correctness", type=bool, default=False, required=False)
    argparser.add_argument("--truncateInput", help="Number of lines to truncate the input files to when parsing", type=int, default=0, required=False)
    argparser.add_argument("--kaggle", help="generate kaggle submission", type=bool, default=0, required=False)
    args = argparser.parse_args()

    train_data = pd.read_csv('Data/train.csv', index_col=0)
    test_data = pd.read_csv('Data/test.csv', index_col=0)
    questions_data = pd.read_csv('Data/questions_ner.csv', index_col=0)

    if (args.truncateInput > 0):
        train_data = train_data[:args.truncateInput]
        test_data = test_data[:args.truncateInput]
        #questions_data = questions_data[:args.truncateInput]

    questions_data, categories = prepareQuestionData(questions_data)
    train_X, train_y = prepareTrainingData(train_data, questions_data)
    test_X = prepareTestData(test_data, questions_data)
    
#     avgBuzzPerUserByCategory(train_X, train_y)
#     exit()
    
    # Split the training set into dev_test and dev_train
    x_train, x_test, y_train, y_test = train_test_split(train_X.as_matrix(), train_y.as_matrix(), train_size=args.subsample*0.75, test_size=args.subsample*0.25, random_state=int(random.random()*100))

    #make y's floats so LR doesnt think there are many classes
    widths = getFeatureColumnWidths(x_train)
    x_train = reshapeFeatureVector(x_train, widths)
    x_test = reshapeFeatureVector(x_test, widths)

    # Train LogisticRegression Classifier
    print "Training regression"
    #logReg = LogisticRegression(C=args.cVal, penalty=args.penalty)
    logReg = linear_model.Lasso(alpha = 0.1)
    logReg.fit(x_train, abs(y_train) if args.twoStage else y_train)

    print "Performing regression"
    y_test_predict = logReg.predict(x_test)
    y_train_predict = logReg.predict(x_train)

    svmCorrect = None
    if args.twoStage:
        #add the buzz position as a feature for the correctness classifier
        x_train_r2 = appendBuzzPosition(x_train, y_train)
        x_test_r2 = appendBuzzPosition(x_test, y_test_predict)

        print "Performing second stage training"
        #logregCorrect = LogisticRegression(C=1, penalty=args.penalty, tol=0.01)#, random_state=random.randint(0,1000000))
        #logregCorrect.fit_transform(x_train_r2, sign(y_train))

        class_Weights = {1:1,-1:2.5}
        svmCorrect = SVC(C=1, class_weight=class_Weights)
        svmCorrect.fit(x_train_r2, sign(y_train))

        print "Performing second stage prediction"
        #y_test_predict_r2 = logregCorrect.predict(x_test_r2)
        y_test_predict_r2 = svmCorrect.predict(x_test_r2)
        y_train_predict_r2 = svmCorrect.predict(x_train_r2)

        for j in range(len(y_test_predict)):
            y_test_predict[j] = sign(y_test_predict_r2[j]) * y_test_predict[j]

        for j in range(len(y_train_predict_r2)):
            y_train_predict_r2[j] *= abs(y_train[j])


    print "-----Begin Analysis------"
    
    userQuestionIntersection(train_X, test_X)
    
    print "Training data analysis:"
    doAnalysis(y_train_predict, y_train, x_train, categories)

    print "Test data analysis:"
    doAnalysis(y_test_predict,y_test,x_test,categories)
    
    print "------End Analysis-------"


    if args.kaggle:
        '''Now re-fit the Model on the full data set'''
        # re-train on dev_test + dev_train
        train_y = train_y.as_matrix().ravel()
        print "Training first-stage guess model"
        logReg.fit_transform(train_X, abs(train_y) if args.twoStage else train_y)

        # Get predictions
        print "Performing first stage guess"
        predictions = logReg.predict(test_X)

        if (args.twoStage):
            print "Preparing second stage data"
            train_X = appendBuzzPosition(train_X, train_y)
            test_X = appendBuzzPosition(test_X, predictions)

            print "Training second stage guess model"
            svmCorrect.fit(train_X, sign(train_y))

            print "Performing second stage guess"
            predictions_2 = svmCorrect.predict(test_X)
            for i in range(len(predictions_2)):
                predictions[i] *= sign(predictions_2[i])

        print "Writing results of guess"
        test_data["position"] = predictions
        test_data.to_csv("Data/guess.csv")
