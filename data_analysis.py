import argparse
import pandas as pd
import numpy as np
import random
import ast
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
#Seed the random so that we can compare code changes
random.seed(12345)

def prepareTrainingData(train_data, questions_data):
    print "Preparing Training Data"
    # Parse the tokenized text as a dictionary and set the values as the keys (bag-of-words)
    questions_data.tokenized_text = questions_data.tokenized_text.map(lambda x:{key:value for (key, value) in (ast.literal_eval(x)).iteritems()})
    # Get the categories we're working with:
    categories = pd.Series(questions_data[["category"]].values.ravel()).unique()
    # Replace the category name with it's corresponding index
    questions_data.category = questions_data.category.map(lambda x:np.where(categories == x)[0][0])
    # Get the text length for all the questions
    questions_data["text_length"] = questions_data.tokenized_text.map(lambda x:len(x))

    # Build the training set
    train = pd.merge(right=questions_data, left=train_data, left_on="question", right_index=True)
    train_X = train[['user', 'text_length', 'category', 'question']]
    # dicts = list(x[0] for x in train_X[['tokenized_text']].values)
    # v = DictVectorizer()
    # v.fit_transform(dicts)
    train_y = train[['position']]
    
    return train_X, train_y, categories


def calcRMS(prediction, actual):
    x = 0.0
    for i in range(0,len(actual)):
            x += (prediction[i] - actual[i])**2
    x /= (len(actual))
    return math.sqrt(x)

def signMismatchPercentage(prediction,actual):
    x = 0.0
    for i in range(0,len(actual)):
        x += (sign(prediction[i]) == sign(actual[i]))    
    return x*100/len(actual)

def calcRMSPerCategory(prediction,actual,features,categories):
    print ("%15s%9s%18s"%("Category","RMS","+/- Accuracy"))
    for index, category in enumerate(categories):
        questionIndices = [i for i in range(0,len(actual)) if (features[i][2] == index)]
        predictionsOfCategory = prediction[questionIndices]
        actualsOfCategory = actual[questionIndices]
        rmsForCategory = calcRMS(predictionsOfCategory,actualsOfCategory)
        signAccuracy = signMismatchPercentage(predictionsOfCategory,actualsOfCategory)
        print ("%15s%10.2f%12.2f%%" % (category,rmsForCategory,signAccuracy))
        

def doAnalysis(prediction, actual,features,categories):
    print "------------Analysis---------------"
    absRms = calcRMS(abs(prediction),abs(actual))
    print ("Ignoring correctness of answer, obtained an RMS of: %.2f" % absRms)
    rms = calcRMS(prediction,actual)
    print ("Obtained a true RMS of: %.2f" % rms)
    signAccuracy = signMismatchPercentage(prediction,actual)
    print ("Predicted Correct Sign %.2f%% of the time." % signAccuracy)
    
    calcRMSPerCategory(prediction,actual,features,categories)

    print "------------End Analysis-----------"


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--subsample", help="Percentage of the dataset to use", type=float, default=.2, required=False)
    argparser.add_argument("--penalty", help="Penalty Function for L.R. (l1 or l2)", type=str, default="l1", required=False)
    argparser.add_argument("--cVal", help="C value for L.R.", type=float, default=10, required=False)
    argparser.add_argument("--twoStage", help="Splits the classification into two stages, position and correctness", type=bool, default=True, required=False)
    args = argparser.parse_args()

    train_data = pd.read_csv('Data/train.csv', index_col=0)
    test_data = pd.read_csv('Data/test.csv', index_col=0)
    questions_data = pd.read_csv('Data/questions.csv', index_col=0)

    train_X, train_y,categories = prepareTrainingData(train_data, questions_data)

    # Split the training set into dev_test and dev_train
    x_train, x_test, y_train, y_test = train_test_split(train_X, train_y, train_size=args.subsample*0.75, test_size=args.subsample*0.25, random_state=int(random.random()*100))

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # Train LogisticRegression Classifier
    print "Performing regression"
    logReg = LogisticRegression(C=args.cVal, penalty=args.penalty, tol=0.01)#,random_state=random.randint(0,1000000))
    logReg.fit_transform(x_train, abs(y_train) if args.twoStage else y_train)
    coef = logReg.coef_.ravel()
    sparsity = np.mean(coef == 0) * 100
    print("C=%.2f" % args.cVal)
    print("Sparsity with %s penalty: %.2f%%" % (args.penalty,sparsity))
    print("score with %s penalty: %.4f" % (args.penalty,logReg.score(x_test, y_test)))
     
    y_test_predict = logReg.predict(x_test)
    
    #add the buzz position as a feature for the correctness classifier
    y_test_predict_r2 = []
    x_train_r2 = np.zeros((len(x_train),5))
    x_test_r2 = np.zeros((len(x_test),5))

    logregCorrect = None
    if args.twoStage:
        print "Performing second stage fit"
        for i in range(len(x_train)):
            x_train_r2[i] = np.append(x_train[i], abs(y_train[i]))
        
        #logregCorrect = LogisticRegression(C=1, penalty=args.penalty, tol=0.01)#, random_state=random.randint(0,1000000))
        #logregCorrect.fit_transform(x_train_r2, sign(y_train))
        
        class_Weights = {1:1,-1:2.5}
        svmCorrect = SVC(C=1, class_weight=class_Weights)
        svmCorrect.fit(x_train_r2, sign(y_train))
        
        for i in range(len(x_test)):
            x_test_r2[i] = np.append(x_test[i], y_test_predict[i])
            
        print "Performing second stage prediction"
        #y_test_predict_r2 = logregCorrect.predict(x_test_r2)
        y_test_predict_r2 = svmCorrect.predict(x_test_r2)
        y_train_predict_r2 = svmCorrect.predict(x_train_r2)
        
        for j in range(len(y_test_predict)):
            y_test_predict[j] = sign(y_test_predict_r2[j]) * y_test_predict[j]
            
        for j in range(len(y_train_predict_r2)):
            y_train_predict_r2[j] *= abs(y_train[j])
            
        print "Training data analysis:"
        doAnalysis(y_train_predict_r2, y_train)
    
    print "Test data analysis:"
    doAnalysis(y_test_predict,y_test,x_test,categories)
    
    
    """ Now re-fit the Model on the full data set """
    # re-train on dev_test + dev_train
    logReg.fit_transform(train_X, train_y.as_matrix(["position"]).ravel())

    # Build the test set
    test = pd.merge(right=questions_data, left=test_data, left_on="question", right_index=True)
    test = test[['user', 'text_length', 'category', 'question']]

    # Get predictions
    predictions = logReg.predict(test)
    test_data["position"] = predictions
    test_data.to_csv("Data/guess.csv")

