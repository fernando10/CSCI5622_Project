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
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
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
    # Parse the tokenized text as a dictionary
    questions_data.tokenized_text = questions_data.tokenized_text.map(lambda x:ast.literal_eval(x))
    # Get the categories we're working with:
    categories = pd.Series(questions_data[["category"]].values.ravel()).unique()
    # Replace the category name with it's corresponding index
    questions_data.category = questions_data.category.map(lambda x:np.where(categories == x)[0][0])
    # Get the text length for all the questions
    questions_data["text_length"] = questions_data.tokenized_text.map(lambda x:len(x))
    # Process the NER columns (convert to dictionary)
    questions_data.DATE = questions_data.DATE.map(lambda x: ast.literal_eval(x))
    questions_data.LOCATION = questions_data.LOCATION.map(lambda x: ast.literal_eval(x))
    questions_data.MONEY = questions_data.MONEY.map(lambda x: ast.literal_eval(x))
    questions_data.ORGANIZATION = questions_data.ORGANIZATION.map(lambda x: ast.literal_eval(x))
    questions_data.PERCENT = questions_data.PERCENT.map(lambda x: ast.literal_eval(x))
    questions_data.PERSON = questions_data.PERSON.map(lambda x: ast.literal_eval(x))
    questions_data.TIME = questions_data.TIME.map(lambda x: ast.literal_eval(x))
    #entities = questions_data[['DATE', 'MONEY', 'ORGANIZATION', 'PERCENT', PERSON, TIME"]]
    #questions_data.accinfo = questions_data.apply(buildAccumulatedQuestionInformation, axis=1)

    #questions_data.ner = questions_data.ner.map(buildWordCategoryFeatures)

    # Build the training set
    train = pd.merge(right=questions_data, left=train_data, left_on="question", right_index=True)
    train_X = train[['user', 'text_length', 'category', 'question', 'DATE', 'LOCATION', 'MONEY', 'ORGANIZATION', 'PERCENT', 'PERSON', 'TIME']]
    # dicts = list(x[0] for x in train_X[['tokenized_text']].values)
    # v = DictVectorizer()
    # v.fit_transform(dicts)
    train_y = train[['position']]

    return train_X, train_y

# def buildAccumulatedQuestionInformation(row):
#     newFeatureVector = []
#     multiplier = 0.1
#     for index, item in enumerate(row.tokenized_text.iteritems()):
#         newFeatureVector.append[multiplier*]
#     return 0

def processNERData(label, index_input):
    indexes = ast.literal_eval(index_input)
    counter = 0
    ret = {}
    for ii, word_position in enumerate(indexes):
        # Skip contiguous indexes, probably the same work tagged several times, like "Albert\PERSON Einstein\PERSON"
        if (ii > 0 and indexes[ii-1] == word_position):
            continue

        ret[label + str(counter)] = word_position

    return ret

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


def reshapeFeatureVector(train_X, columnWidths):
    newFeatureMatrix = np.zeros((train_X.shape[0], sum(columnWidths)))
    for rowIndex, featureVector in enumerate(train_X):
        newColumnIndex = 0
        for origColIndex, colWidth in enumerate(columnWidths):
            if colWidth == 1:
                newFeatureMatrix[rowIndex][newColumnIndex] = featureVector[origColIndex]
            else:
                for ii in range(int(colWidth)):
                    if (ii < len(featureVector[origColIndex])):
                        newFeatureMatrix[rowIndex][newColumnIndex + ii] = featureVector[origColIndex][ii]
            
            newColumnIndex += colWidth
    
    return newFeatureMatrix

def flattenFeaturesIntoColumns(train_X):
    columnWidths = np.zeros(train_X.shape[1])

    for row in train_X:
        for index, col in enumerate(row):
            length = getLengthOfFeature(col)
            if columnWidths[index] < length:
                columnWidths[index] = length;
    return columnWidths

def getLengthOfFeature(feature):
    try:
        return len(feature)
    except TypeError:
        return 1

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
    args = argparser.parse_args()

    train_data = pd.read_csv('Data/train.csv', index_col=0)
    test_data = pd.read_csv('Data/test.csv', index_col=0)
    questions_data = pd.read_csv('Data/questions_ner.csv', index_col=0)

    train_X, train_y = prepareTrainingData(train_data, questions_data)

    # Split the training set into dev_test and dev_train
    x_train, x_test, y_train, y_test = train_test_split(train_X, train_y, train_size=args.subsample*0.75, test_size=args.subsample*0.25, random_state=int(random.random()*100))

    x_train = flattenFeaturesIntoColumns(x_train)
    x_test = flattenFeaturesIntoColumns(x_test)
    # Train Regression
    print "Performing regression"

    reg = SVR(kernel='rbf', C=1e3, gamma=0.1)
    #reg = DecisionTreeRegressor(max_depth=5)
    y_train = np.array([x[0] for x in y_train])
    reg.fit(x_train, y_train.ravel())

    y_test_predict = reg.predict(x_test)

    # add the buzz position as a feature for the correctness classifier
    # y_test_predict_r2 = []
    # x_train_r2 = np.zeros((len(x_train),len(x_train[0]) + 1))
    # x_test_r2 = np.zeros((len(x_test),len(x_train[0]) + 1))

    # # logregCorrect = None
    # if args.twoStage:
    #     print "Performing second stage fit"
    #     for i in range(len(x_train)):
    #         x_train_r2[i] = np.append(x_train[i], abs(y_train[i]))

    #     #logregCorrect = LogisticRegression(C=1, penalty=args.penalty, tol=0.01)#, random_state=random.randint(0,1000000))
    #     #logregCorrect.fit_transform(x_train_r2, sign(y_train))

    #     class_Weights = {1:1,-1:2.5}
    #     svmCorrect = SVC(C=1, class_weight=class_Weights)
    #     svmCorrect.fit(x_train_r2, sign(y_train))

    #     for i in range(len(x_test)):
    #         x_test_r2[i] = np.append(x_test[i], y_test_predict[i])

    #     print "Performing second stage prediction"
    #     #y_test_predict_r2 = logregCorrect.predict(x_test_r2)
    #     y_test_predict_r2 = svmCorrect.predict(x_test_r2)
    #     y_train_predict_r2 = svmCorrect.predict(x_train_r2)

    #     for j in range(len(y_test_predict)):
    #         y_test_predict[j] = sign(y_test_predict_r2[j]) * y_test_predict[j]

    #     for j in range(len(y_train_predict_r2)):
    #         y_train_predict_r2[j] *= abs(y_train[j])

    #     print "Training data analysis:"
    #     doAnalysis(y_train_predict_r2, y_train)

    #print "Test data analysis:"
    doAnalysis(y_test_predict,y_test)


    # Now re-fit the Model on the full data set
    # re-train on dev_test + dev_train
    # print "Training Regression"
    # #pdb.set_trace()
    # train_y = np.array([x[0] for x in train_y.as_matrix()])
    # reg.fit(train_X.as_matrix(), train_y.ravel())

    # # # Build the test set
    # test = pd.merge(right=questions_data, left=test_data, left_on="question", right_index=True)
    # test = test[['user', 'text_length', 'category', 'question']]

    # # # Get predictions
    # predictions = reg.predict(test)
    # test_data["position"] = predictions
    # test_data.to_csv("Data/guess.csv")

