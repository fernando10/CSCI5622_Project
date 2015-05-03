import pdb
import ast
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from test.test_math import acc_check


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


def prepareQuestionData(questions_data):
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
    return questions_data, categories

def addFeaturesForUserAccuracyAndPositionAverage(x, usersResponseData):
    if x.user in usersResponseData:
        userInfo = usersResponseData[x.user]
        x["overall_accuracy"] = userInfo["overall_acc"]
        x["overall_position"] = userInfo["overall_pos"]
        x["category_accuracy"]= userInfo[str(x.category)+"_acc"]
        x["category_position"]= userInfo[str(x.category)+"_pos"]
    else:
        x["overall_accuracy"] = .73
        x["overall_position"] = 80
        x["category_accuracy"]= categoryAccAverages[x.category]
        x["category_position"]= categoryPosAverages[x.category]
    return x

categoryPosAverages = None
categoryAccAverages = None
def generateUserResponseData(train):
    prelimUserResponseData = defaultdict(lambda :defaultdict(int))
    for key,row in train.iterrows():
        prelimUserResponseData[row.user][str(row.category) + "_acc"] += 1 if row.position > 0 else 0
        prelimUserResponseData[row.user][str(row.category) + "_pos"] += abs(row.position)
        prelimUserResponseData[row.user][str(row.category) + "_count"] += 1
    global categoryPosAverages
    global categoryAccAverages
    categoryPosAverages = {}
    categoryAccAverages = {}
    train['correct'] = train['position'].map(lambda x: 0 if x < 0 else x)
    for ii in range(0,11):
        categoryPosAverages[ii] = train.groupby('category').agg({'position': lambda x : np.mean(abs(x))})['position'][ii]
        categoryAccAverages[ii] = train.groupby('category').agg({'correct': np.mean})['correct'][ii]
       
    userResponseData = defaultdict(dict)
    for user, data in prelimUserResponseData.iteritems():
        totalAcc = 0
        totalPos = 0
        count = 0
        for i in range(0, 11):
            c = data.get(str(i) + "_count", 0)
            acc = data.get(str(i) + "_acc", 0)
            pos = data.get(str(i) + "_pos", 0)
            userResponseData[user][str(i) + "_acc"] = (float(acc) / c) if c > 0 else categoryAccAverages[i]
            userResponseData[user][str(i) + "_pos"] = (float(pos) / c) if c > 0 else categoryPosAverages[i]
            totalAcc += acc
            totalPos += pos
            count += c
        
        userResponseData[user]["overall_acc"] = float(totalAcc) / count
        userResponseData[user]["overall_pos"] = float(totalPos) / count
    
    return userResponseData

userResponseData = None
def prepareTrainingData(train_data, questions_data):
    print "Preparing Training Data"
    # Build the training set
    train = pd.merge(right=questions_data, left=train_data, left_on="question", right_index=True)

    global userResponseData
    userResponseData = generateUserResponseData(train) 
    
    train = train.apply(lambda x: addFeaturesForUserAccuracyAndPositionAverage(x,userResponseData),axis=1)
    train_X = train.drop(['answer_x', 'answer_y', 'set', 'raw_text', 'tokenized_text', 'position','correct'], 1)
    train_y = train[['position']]

    return train_X, train_y

def prepareTestData(test_data, questions_data):
    print "Preparing Test Data"
    # Build the test set
    test = pd.merge(right=questions_data, left=test_data, left_on="question", right_index=True)
    test = test.apply(lambda x: addFeaturesForUserAccuracyAndPositionAverage(x,userResponseData),axis=1)
    
    test_X = test.drop(['answer_x', 'answer_y', 'set', 'raw_text', 'tokenized_text', 'position'], 1)

    return test_X, test

