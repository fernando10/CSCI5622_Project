


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--subsample", help="Percentage of the dataset to use", type=float, default=.2, required=False)
    argparser.add_argument("--kaggle" help="Output a guess.csv file for kaggle" type=bool, default=false, required=False)
    args = argparser.parse_args()
