import os.path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def explore():
    df = pd.read_csv("Results/feature_vector.csv")
    features = df.columns[:-1]
    fig, axs = plt.subplots(4,3, figsize=(12,9))
    axs = axs.flatten()
    palette = ["limegreen", "dimgray"]
    for i, feature in enumerate(features):
        ax = axs[i]
        sns.kdeplot(df, x=feature, hue="Label", hue_order=["positive", "negative"], palette=palette, ax=ax)
        ax.set_title(feature)
        ax.set(xlabel=None, ylabel=None)
        ax.set_xlim([0, 1])
        if i != 2:
            ax.get_legend().remove()
        ax.set_yticks([])
    fig.tight_layout()

    if not os.path.exists("Results"):
        os.makedirs("Results")
    plt.savefig("Results/feature_distribution.png")
