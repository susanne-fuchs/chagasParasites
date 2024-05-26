import glob
import matplotlib.image as mpimg
import numpy as np
import pandas as pd


def create_feature_vectors():
    """
    Split each image into 3 components and store the min, max and mean of each component in a data frame.
    :return:
    """
    columns = ["R_min", "G_min", "B_min",
               "R_max", "G_max", "B_max",
               "R_mean", "G_mean", "B_mean",
               "R_median", "G_median", "B_median",
               "Label"]
    df = pd.DataFrame(columns=columns)
    for label in ["positive", "negative"]:
        image_files = glob.glob(f"Data/{label}/*.png")
        for image_file in image_files:
            img = mpimg.imread(image_file)
            # calculate min, max and mean of the three color components r, g, b.

            mins = np.min(img, (0, 1)).tolist()
            maxs = np.max(img, (0, 1)).tolist()
            means = np.mean(img, (0, 1)).tolist()
            medians = np.median(img, (0, 1)).tolist()
            row = pd.DataFrame([mins + maxs + means + medians + [label]], columns=columns)
            df = pd.concat([df if not df.empty else None, row], ignore_index=True)
    df.to_csv("Data/feature_vector.csv", index=False)