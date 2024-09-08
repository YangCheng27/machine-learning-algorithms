import sklearn.datasets as sk_datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    cal_housing = sk_datasets.fetch_california_housing(as_frame = True)
    df = cal_housing.frame
    df.hist(bins=50, figsize=(8,6), color='salmon')
    plt.tight_layout()
    return df

def correlation_map(df):
    attributes = df.columns.tolist()

    fig, axes = plt.subplots(len(attributes), len(attributes), figsize=(16, 12), layout='constrained')

    for i in range(len(attributes)):
        for j in range(len(attributes)):
            if i > j:
                axes[i, j].scatter(df[attributes[j]], df[attributes[i]], alpha=0.25)
                if j == 0:
                    axes[i, j].set_ylabel(attributes[i])
                if i == len(attributes) - 1:
                    axes[i, j].set_xlabel(attributes[j])
            else:
                axes[i, j].set_visible(False)
    fig.align_labels()
    plt.tight_layout()
    plt.show()

def correlation_map3(df):

    corr_matrix = df.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.show()



