import argparse
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pdb

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--subsample", help="Percentage of the dataset to use", type=float, default=1.0, required=False)
    args = argparser.parse_args()

    train_data = pd.read_csv('Data/train.csv', index_col=0)
    test_data = pd.read_csv('Data/test.csv', index_col=0)
    questions_data = pd.read_csv('Data/questions.csv', index_col=0)

    # Parse the tokenized text as a dictionary
    questions_data.tokenized_text = questions_data.tokenized_text.map(lambda x: ast.literal_eval(x))

    # Get the categories we're working with:
    categories = pd.Series(questions_data[["category"]].values.ravel()).unique()

    # Replace the category name with it's corresponding index
    questions_data.category = questions_data.category.map(lambda x: np.where(categories == x)[0][0])

    # Build the training set
    train = pd.merge(right=questions_data, left=train_data, left_on="question", right_index=True)
    train_X = train[['user', 'tokenized_text', 'category']]
    train_y = train[['position']]

    # Split the training set into dev_test and dev_train
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, train_size=args.subsample*0.75, test_size=args.subsample*0.25, random_state=int(random.random()*100))

    # Build the test set
    test = pd.merge(right=questions_data, left=test_data, left_on="question", right_index=True)
    test = test[['user', 'tokenized_text', 'category']]
