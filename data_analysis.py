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

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--subsample", help="Percentage of the dataset to use", type=float, default=1.0, required=False)
    args = argparser.parse_args()

    train_data = pd.read_csv('Data/train.csv', index_col=0)
    test_data = pd.read_csv('Data/test.csv', index_col=0)
    questions_data = pd.read_csv('Data/questions.csv', index_col=0)

    # Parse the tokenized text as a dictionary and set the values as the keys (bag-of-words)
    questions_data.tokenized_text = questions_data.tokenized_text.map(lambda x: {key:value for key,value in (ast.literal_eval(x)).iteritems()})

    # Get the categories we're working with:
    categories = pd.Series(questions_data[["category"]].values.ravel()).unique()

    # Replace the category name with it's corresponding index
    questions_data.category = questions_data.category.map(lambda x: np.where(categories == x)[0][0])

    # Get the text length for all the questions
    questions_data["text_length"] = questions_data.tokenized_text.map(lambda x: len(x))

    # Build the training set
    train = pd.merge(right=questions_data, left=train_data, left_on="question", right_index=True)

    train_X = train[['user', 'text_length', 'category', 'question']]
    # dicts = list(x[0] for x in train_X[['tokenized_text']].values)
    # v = DictVectorizer()
    # v.fit_transform(dicts)
    train_y = train[['position']]

    # Regularize the training set
    #train_X = StandardScaler().fit_transform(train_X)

    # Split the training set into dev_test and dev_train
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, train_size=args.subsample*0.75, test_size=args.subsample*0.25, random_state=int(random.random()*100))

    # Train LogisticRegression Classifier
    C = 10
    logReg = LogisticRegression(C=C, penalty='l1', tol=0.01)
    logReg.fit_transform(X_train, y_train.ravel())
    coef = logReg.coef_.ravel()
    sparsity = np.mean(coef == 0) * 100
    print("C=%.2f" % C)
    print("Sparsity with L1 penalty: %.2f%%" % sparsity)
    print("score with L1penalty: %.4f" % logReg.score(X_test, y_test.ravel()))

    # re-train on dev_test + dev_train
    logReg.fit_transform(train_X, train_y)


    # Build the test set
    test = pd.merge(right=questions_data, left=test_data, left_on="question", right_index=True)
    test = test[['user', 'text_length', 'category', 'question']]

    # Get predictions
    predictions = logReg.predict(test)
    test_data["position"] = predictions
    test_data.to_csv("guess.csv")

