import ast
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction import DictVectorizer


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

def prepareXData(train_data, questions_data):
    train = pd.merge(right=questions_data, left=train_data, left_on="question", right_index=True)
    train_X = train[['user', 'text_length', 'category', 'question', 'DATE', 'LOCATION', 'MONEY', 'ORGANIZATION', 'PERCENT', 'PERSON', 'TIME']]

    return train_X, train

def prepareTrainingData(train_data, questions_data):
    print "Preparing Training Data"
    # Build the training set
    train_X, train = prepareXData(train_data, questions_data)
    train_y = train[['position']]

    return train_X, train_y

def prepareTestData(test_data, questions_data):
    print "Preparing Test Data"
    # Build the test set

    test_X, train = prepareXData(test_data, questions_data)

    return test_X

