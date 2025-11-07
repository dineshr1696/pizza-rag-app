import pandas as pd

def load_reviews():
    path = "/content/project/restaurant_reviews.csv"
    df = pd.read_csv(path)
    return df
