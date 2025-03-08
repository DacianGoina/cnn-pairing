"""
Data load utis - import and split
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import copy
from sklearn.datasets import fetch_openml
from copy import deepcopy
from transformation_utils import reshape_to_single_channel


# Load MNIST fashion or digits dataset
def load_MNIST_data(dataset_name = "digits"):
  if dataset_name == "fashion":
    mnist = fetch_openml('mnist_784', version=1)  # this is CIFAR
    X_origin, y_origin = mnist.data, mnist.target

    X = deepcopy(X_origin)
    y = deepcopy(y_origin)


  if dataset_name == "digits":
    mnist = fetch_openml('fashion-mnist', version=1)
    X_origin, y_origin = mnist.data, mnist.target

    X = deepcopy(X_origin)
    y = deepcopy(y_origin)



def import_and_split_data(X_origin, y_origin, used_data_percentage, split_seed,  split_percentage_test, split2_seed, rows_num, cols_num, no_of_classes_v):
  X = deepcopy(X_origin)
  y = deepcopy(y_origin)

  # convert output values to int
  y = y.astype(int)

  #Step 2: Take just a fraction (e.g 50%) from all dataset, e.g do not use all data for experiment
  X_sample = None
  y_sample = None

  if used_data_percentage == 1:
    # use whole data
    X_sample, y_sample = X, y
  else:
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size= used_data_percentage, stratify=y, random_state=split_seed)

  # with the selected data, split for train and test
  x_train, x_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=split_percentage_test, stratify=y_sample, random_state= split2_seed)

  # from pd df to numpy arrays
  x_train = x_train.values
  x_test = x_test.values

  y_train = y_train.values
  y_test = y_test.values

  y_train_distribution = get_occurrences_of_unique_vals(y_train)
  y_test_distribution = get_occurrences_of_unique_vals(y_test)

  # convert to matrices of single channel
  x_train, x_test, y_train, y_test = reshape_to_single_channel(x_train, x_test, y_train, y_test, rows_num, cols_num, no_of_classes = no_of_classes_v)
  return x_train, x_test, y_train, y_test, y_train_distribution, y_test_distribution

def get_occurrences_of_unique_vals(y):
  unique_values, counts = np.unique(y, return_counts=True)

  # Combine the unique values with their counts
  occurrences = dict(zip(unique_values, counts))

  return occurrences