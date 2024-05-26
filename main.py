from preprocessing import create_feature_vectors
from explore import explore
from neural_networks import create_datasets_and_train_model


if __name__ == '__main__':
    # create_feature_vectors()
    # explore()
    # Blue minimum is mostly the same for negative and positive images.
    # Mean and median and maximum behave mostly the same.
    # Therefore, I include
    # R_min, G_min, R_max, G_max, B_max, R_mean, G_mean and B_mean
    # in the feature vector.
    create_datasets_and_train_model()