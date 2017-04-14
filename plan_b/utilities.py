import numpy as np

def generate_random_subset(df, subset_size):
    n = len(df)
    indexes = np.arange(n)
    np.random.shuffle(indexes)

    if subset_size <= 1: m = int(subset_size*n)

    np.sort(indexes)

    return df.iloc[indexes[:m],:]