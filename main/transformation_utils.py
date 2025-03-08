"""
A class that contains utilities function to resize images across channels and dimensions
"""

import numpy as np
import torch
import torch.nn.functional as F
import tensorflow
from tensorflow.keras.utils import to_categorical

# Reshape to add a single channel dimension (28x28x1) and normalize pixel values
# e.g. reshape_to_single_channel(x_train, x_test, y_train, y_test, rows_num = 28, cols_num = 28, no_of_classes = 10)
def reshape_to_single_channel(x_train, x_test, y_train, y_test, rows_num, cols_num, no_of_classes = 10):
  x_train = x_train.reshape(-1, rows_num, cols_num, 1).astype('float32')
  x_test = x_test.reshape(-1, rows_num, cols_num, 1).astype('float32')

  # One-hot encode the labels
  y_train = to_categorical(y_train, no_of_classes)
  y_test = to_categorical(y_test, no_of_classes)

  return x_train, x_test, y_train, y_test

# e.g. in (28, 28) out (28, 14)
def resize_2d_np(x_item, new_size, mode_val, normalize = True):
  # mode can be 'bicubic' 'nearest'
  image_tensor = torch.tensor(x_item).unsqueeze(0).unsqueeze(0).float()
  resized_image_tensor = F.interpolate(image_tensor, size=new_size, mode=mode_val)
  if normalize == False:
    resized_image_tensor = resized_image_tensor.squeeze(0).squeeze(0).detach().numpy()
    return resized_image_tensor

  min_val = resized_image_tensor.min()
  max_val = resized_image_tensor.max()

# Scale the tensor to [0, 1] using the formula
  normalized_image_tensor = (resized_image_tensor - min_val) / (max_val - min_val)
  normalized_image_tensor = normalized_image_tensor.squeeze(0).squeeze(0).detach().numpy()
  return normalized_image_tensor

# e.g. in : (5000, 28, 28), out: (5000, 28, 14)
def resize_list_2d_np(x_list, new_size, mode_val, normalize = True):
  res = [resize_2d_np(item, new_size, mode_val, normalize) for item in x_list]
  return  np.asarray(res)

# convert e.g (28, 28, 1) to (28, 28)
def threeD_one_ch_to_2d(input_x):
  output_list = [item.squeeze() for item in input_x]
  return np.asarray(output_list)

# e.g (28, 28) to (28 * 28)
def flat_to_1d(input_x):
  output_list = [item.flatten() for item in input_x]
  return np.asarray(output_list)

# convert e.g (28, 28) to (28, 28, 1)
def twoD_to_threeD_one_ch(input_x):
  output_list = [item.reshape(item.shape[0], item.shape[1], 1) for item in input_x]
  return np.asarray(output_list)


def threeD_oneCh_to_threeD_oneCh_resize_with_F(input_x, new_size, mode_val):
  as_two_dim_train = threeD_one_ch_to_2d(input_x)
  resized = resize_list_2d_np(as_two_dim_train, new_size, mode_val)
  fin_3d_one_ch = twoD_to_threeD_one_ch(resized)
  return fin_3d_one_ch

# Min max scaling to [0, 1], in: np array
def reduce_by_max(input_x):
  max_value = np.max(input_x)
  return input_x / max_value
