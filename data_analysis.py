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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
import pdb
import math
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
    
    questions_data.ner = questions_data.ner.map(buildWordCategoryFeatures)
    
    # Build the training set
    train = pd.merge(right=questions_data, left=train_data, left_on="question", right_index=True)
    train_X = train[['user', 'text_length', 'category', 'question','ner']]
    # dicts = list(x[0] for x in train_X[['tokenized_text']].values)
    # v = DictVectorizer()
    # v.fit_transform(dicts)
    train_y = train[['position']]
    
    return train_X, train_y

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


def calcRMS(prediction, actual, absvals=False):
    x = 0.0
    for i in range(0,len(actual)):
        if absvals:
            x += (abs(prediction[i]) - abs(actual[i]))**2
        else:
            x += (prediction[i] - actual[i])**2
    x /= (len(actual))
    return math.sqrt(x)

def signMismatchPercentage(prediction,actual):
    x = 0.0
    for i in range(0,len(actual)):
        x += (sign(prediction[i]) == sign(actual[i]))    
    return x*100/len(actual)

def doAnalysis(prediction, actual):
    print "-------Analysis----------"
    absRms = calcRMS(prediction,actual,True)
    print ("Ignoring correctness of answer, obtained an RMS of: %.2f" % absRms)
    rms = calcRMS(prediction,actual)
    print ("Obtained a true RMS of: %.2f" % rms)
    signAccuracy = signMismatchPercentage(prediction,actual)
    print ("Predicted Correct Sign %.2f%% of the time." % signAccuracy)
    print "-------End Analysis------"


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--subsample", help="Percentage of the dataset to use", type=float, default=.2, required=False)
    argparser.add_argument("--penalty", help="Penalty Function for L.R. (l1 or l2)", type=str, default="l2", required=False)
    argparser.add_argument("--cVal", help="C value for L.R.", type=float, default=10, required=False)
    argparser.add_argument("--twoStage", help="Splits the classification into two stages, position and correctness", type=bool, default=True, required=False)
    argparser.add_argument("--truncateInput", help="Number of lines to truncate the input files to when parsing", type=int, default=10, required=False)
    args = argparser.parse_args()

    train_data = pd.read_csv('Data/train.csv', index_col=0)
    test_data = pd.read_csv('Data/test.csv', index_col=0)
    questions_data = pd.read_csv('Data/questions_ner.csv', index_col=0)

    if (args.truncateInput > 0):
        train_data = train_data[:args.truncateInput]
        test_data = test_data[:args.truncateInput]
        questions_data = questions_data[:args.truncateInput]

    train_X, train_y = prepareTrainingData(train_data, questions_data)
    train_X = train_X.as_matrix()
    temp = np.zeros((len(train_X),len(train_X[0]) + 150))
    for i in range(len(train_X)):
        b = list(train_X[i][0:])
        b.extend(train_X[i][-1])
        temp[i] = np.array(b)
    train_X = temp
    # Split the training set into dev_test and dev_train
    x_train, x_test, y_train, y_test = train_test_split(train_X, train_y, train_size=args.subsample*0.75, test_size=args.subsample*0.25, random_state=int(random.random()*100))
    
    #make y's floats so LR doesnt think there are many classes
    #y_train = y_train.ravel()
    #y_test = y_test.ravel()
    y_train = np.array([float(x) for x in y_train["position"]])
    y_test = np.array([float(x) for x in y_test["position"]])

    
  
    
    
   

    # Train LogisticRegression Classifier
    print "Performing regression"
    logReg = LinearRegression(fit_intercept = True)
    logReg.fit(x_train, abs(y_train) if args.twoStage else y_train,n_jobs=4)
    coef = logReg.coef_.ravel()
#    print("DEBUG: Coefficient matrix is %dx%d" % (len(coef),len(coef[0])))
    sparsity = np.mean(coef == 0) * 100
    print("C=%.2f" % args.cVal)
    print("Sparsity with %s penalty: %.2f%%" % (args.penalty,sparsity))
    print("score with %s penalty: %.4f" % (args.penalty,logReg.score(x_test, y_test)))
     
    y_test_predict = logReg.predict(x_test)
    
    #add the buzz position as a feature for the correctness classifier
    y_test_predict_r2 = []
    x_train_r2 = np.zeros((len(x_train),len(x_train[0]) + 1))
    x_test_r2 = np.zeros((len(x_test),len(x_train[0]) + 1))

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
    doAnalysis(y_test_predict,y_test)
    
    
    """ Now re-fit the Model on the full data set 
    # re-train on dev_test + dev_train
    logReg.fit_transform(train_X, train_y.as_matrix(["position"]).ravel())

    # Build the test set
    test = pd.merge(right=questions_data, left=test_data, left_on="question", right_index=True)
    test = test[['user', 'text_length', 'category', 'question']]

    # Get predictions
    predictions = logReg.predict(test)
    test_data["position"] = predictions
    test_data.to_csv("Data/guess.csv")
    """
