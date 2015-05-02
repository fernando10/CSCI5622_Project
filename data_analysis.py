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
#Seed the random so that we can compare code changes
random.seed(12345)

def prepareQuestionData(questions_data):
    questions_data.tokenized_text = questions_data.tokenized_text.map(lambda x:{key:value for (key, value) in (ast.literal_eval(x)).iteritems()})
    # Get the categories we're working with:
    categories = pd.Series(questions_data[["category"]].values.ravel()).unique()
    # Replace the category name with it's corresponding index
    questions_data.category = questions_data.category.map(lambda x:np.where(categories == x)[0][0])
    # Get the text length for all the questions
    questions_data["text_length"] = questions_data.tokenized_text.map(lambda x:len(x))
    
    questions_data.ner = questions_data.ner.map(buildWordCategoryFeatures)

    return questions_data, categories

def prepareTrainingData(train_data, questions_data):
    print "Preparing Training Data"
    # Build the training set
    
    train = pd.merge(right=questions_data, left=train_data, left_on="question", right_index=True)
    train_X = train[['user', 'text_length', 'category', 'question','ner']]
    train_y = train[['position']]
    
    train_X = train_X.as_matrix()
    x_len = len(train_X[0])
    temp = np.zeros((len(train_X),x_len + len(train_X[0][-1]) -1))
    for i in range(len(train_X)):
        b = list(train_X[i][0:x_len - 1])
        b.extend(train_X[i][-1])
        temp[i] = np.array(b)
    train_X = temp
    
    return train_X, train_y

def prepareTestData(test_data, questions_data):
    print "Preparing Test Data"
    # Build the test set
    
    test = pd.merge(right=questions_data, left=test_data, left_on="question", right_index=True)
    test_X = test[['user', 'text_length', 'category', 'question','ner']]
    
    test_X = test_X.as_matrix()
    x_len = len(test_X[0])
    temp = np.zeros((len(test_X),x_len + len(test_X[0][-1]) -1))
    for i in range(len(test_X)):
        b = list(test_X[i][0:x_len - 1])
        b.extend(test_X[i][-1])
        temp[i] = np.array(b)
    test_X = temp

    return test_X

digitRegex = re.compile("\d+")
def buildWordCategoryFeatures(nerString):
    positionValues = {}
    for i in range(0,150):
        positionValues[i] = (i+1) *.1
    if (isinstance(nerString, str)):
        sentencePosWithPropNouns = digitRegex.findall(nerString)
        
        if sentencePosWithPropNouns != None:
            positions = [int(x) for x in sentencePosWithPropNouns]
            positionValues[0] = .1
            
            for i in range(1,150):
                positionValues[i] = positionValues[i-1] + .1
                if i in positions:
                    positionValues[i] += 1 
                    
    vectorizer = DictVectorizer()
    return vectorizer.fit_transform(positionValues).toarray()[0].tolist()

def appendBuzzPosition(x_data, positions):
    x_data_2 = np.zeros((len(x_data),len(x_data[0]) + 1))
    for i in range(len(x_data_2)):
        x_data_2[i] = np.append(x_data[i], abs(positions[i]))

    return x_data_2

def calcRMS(prediction, actual):
    if (len(actual) == 0):
        return 0
    x = 0.0
    for i in range(0,len(actual)):
            x += (prediction[i] - actual[i])**2
    x /= (len(actual))
    return math.sqrt(x)

def signMismatchPercentage(prediction,actual):
    if (len(actual) == 0):
        return 0
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
    argparser.add_argument("--subsample", help="Percentage of the dataset to use", type=float, default=1, required=False)
    argparser.add_argument("--penalty", help="Penalty Function for L.R. (l1 or l2)", type=str, default="l2", required=False)
    argparser.add_argument("--cVal", help="C value for L.R.", type=float, default=10, required=False)
    argparser.add_argument("--twoStage", help="Splits the classification into two stages, position and correctness", type=bool, default=True, required=False)
    argparser.add_argument("--truncateInput", help="Number of lines to truncate the input files to when parsing", type=int, default=0, required=False)
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
    
    # Split the training set into dev_test and dev_train
    x_train, x_test, y_train, y_test = train_test_split(train_X, train_y, train_size=args.subsample*0.75, test_size=args.subsample*0.25, random_state=int(random.random()*100))
    
    #make y's floats so LR doesnt think there are many classes
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    #y_train = np.array([float(x) for x in y_train["position"]])
    #y_test = np.array([float(x) for x in y_test["position"]])

   

    # Train LogisticRegression Classifier
    print "Training regression"
    logReg = LogisticRegression(C=args.cVal, penalty=args.penalty)
    logReg.fit(x_train, abs(y_train) if args.twoStage else y_train)
    coef = logReg.coef_.ravel()
#    print("DEBUG: Coefficient matrix is %dx%d" % (len(coef),len(coef[0])))
    sparsity = np.mean(coef == 0) * 100
    
    print "Performing regression"
    print("C=%.2f" % args.cVal)
    print("Sparsity with %s penalty: %.2f%%" % (args.penalty,sparsity))
    print("score with %s penalty: %.4f" % (args.penalty,logReg.score(x_test, y_test)))
     
    y_test_predict = logReg.predict(x_test)
    
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
            
        print "Training data analysis:"
        doAnalysis(y_train_predict_r2, y_train, x_train_r2, categories)
    
    print "Test data analysis:"
    doAnalysis(y_test_predict,y_test,x_test,categories)
    
    
    '''Now re-fit the Model on the full data set''' 
    # re-train on dev_test + dev_train
    train_y = train_y.as_matrix().ravel()
    print "Training first-stage guess model"
    logReg.fit_transform(train_X, abs(train_y) if args.twoStage else train_y)

    # Build the test set
    print "Preparing test data"
    test = prepareTestData(test_data, questions_data)

    # Get predictions
    print "Performing first stage guess"
    predictions = logReg.predict(test)
    
    if (args.twoStage):
        print "Preparing second stage data"
        train_X = appendBuzzPosition(train_X, train_y)
        test = appendBuzzPosition(test, predictions)
        
        print "Training second stage guess model"
        svmCorrect.fit(train_X, sign(train_y))
        
        print "Performing second stage guess"
        predictions_2 = svmCorrect.predict(test)
        for i in range(len(predictions_2)):
            predictions[i] *= sign(predictions_2[i])
    
    print "Writing results of guess"
    test_data["position"] = predictions
    test_data.to_csv("Data/guess.csv")
