"""
Cantor pairing function transformation related functions
"""

import numpy as np
import random
from transformation_utils import *


# For a given list a numbers (usually the identity permutation) shuffle it using a seed
def generate_random_pairs(list_of_elems, shuffle_seed = 0):
    # Shuffle the elements randomly - this has side effects

    # for seed 0, do not shuffle
    if shuffle_seed != 0:
      random.Random(shuffle_seed).shuffle(list_of_elems)

    # Create pairs by grouping the elements in twos
    pairs = []
    for i in range(0, len(list_of_elems), 2):
        # If the length is odd, the last element will be left alone
        if i + 1 < len(list_of_elems):
            pairs.append((list_of_elems[i], list_of_elems[i+1]))

    return pairs

# for a given list of integers, generate multiple pairs (a collection for each seed value)
def generate_pairs_instances(list_of_items, seeds = [0]):
  pairs_collection = [generate_random_pairs(list_of_items, shuffle_seed = seed_val) for seed_val in seeds]
  return pairs_collection

# Cantor pairing function
def cantor_pairing(a, b):
  return ((a+b) * (a+b+1) ) * 0.5  + b

# Apply quadratic downscaling (first step for computing the inverse of cantor function) for a value / list of values
def cantor_quadratic_downscaling_faster(x, with_floor= False):
  #vectorized_func = np.vectorize(lambda val: ( (-1 + np.sqrt(1 + 8 * val))/ 4 )  )
  res = (-1 + np.sqrt(1 + 8*x))/4
  if with_floor == True:
    res = np.floor(res)

  return res

# Example usage
# N = 10
# pairs = generate_random_pairs(list(range(0, 100)), shuffle_seed = 0)
# print(pairs)

# Cantor pairing function over a list of values
#  e.g input_x: (1000, 28, 28), pairs: [(0,15), (22, 17), ...]
def cantor_transformation_over_list_of_2d(input_x, pairs):
  result_as_list = [cantor_pairing(input_x[:, index1], input_x[:, index2]) for (index1, index2) in pairs]
  return np.asarray(result_as_list)

# Full Cantor pairing transformation flow - with input, pairs, new row, col size, normalize, quadratic downscaling
# input_x: numpy array of numpy arrays - matrix with one channel each, e.g input_x shape = (50000, 28, 28, 1);
# quadratic_downscale param: tuple (bool, value) to apply or not, and how many times

# usage e.g.
# cantor_transformation_full_flow(x_train, new_rows_size = 28, new_cols_size = 14, used_pairs = generate_random_pairs(list(range(0, 28 * 28)), shuffle_seed = 0))
# cantor_transformation_full_flow(x_train_transformed, new_rows_size = 14, new_cols_size = 7, used_pairs = generate_random_pairs(list(range(0, 14 * 14)), shuffle_seed = 0), normalize = True)
def cantor_transformation_full_flow(input_x, used_pairs, new_rows_size, new_cols_size, normalize = False, quadratic_downscale = (False, 0)):
  as_two_dim_train = threeD_one_ch_to_2d(input_x)  # from (50000, 28, 28, 1) to (50000, 28, 28)
  as_one_dim_train = flat_to_1d(as_two_dim_train) # from (28, 28) to (50000, 28*28)

  input_x_1d_after_mapping = cantor_transformation_over_list_of_2d(as_one_dim_train, used_pairs) # new size is ((28 * 28) /2, 50000) , let 28 * 14
  input_x_1d_after_mapping_tr = input_x_1d_after_mapping.transpose() # transpose, back to (50000, 28 * 14)

  back_to_2d = input_x_1d_after_mapping_tr.reshape(input_x_1d_after_mapping_tr.shape[0], new_rows_size, new_cols_size) # back to (50000, 28, 14) choose 28 to divide (28 * 28)/2, also 28 * 14 = 392
  back_to_3d = twoD_to_threeD_one_ch(back_to_2d) # back to 3d one channel : (50000, 28, 14, 1)

  normalized_data = back_to_3d

  if quadratic_downscale[0] == True:
    for it in range(quadratic_downscale[1]):
      normalized_data = cantor_quadratic_downscaling_faster(normalized_data, with_floor = False)

  if normalize == True:
    normalized_data = reduce_by_max(normalized_data) # move values to [0, 1]

  return normalized_data


# call eg: x_data, [(), (), ....] 28, 14, False so ONLY one time
def cantor_transform_one_unit(x_data, used_pairs, new_rows_size_v, new_cols_size_v, normalize_v, quadratic_downscale_val = (False, 0)):
  x_data_fin = cantor_transformation_full_flow(x_data, used_pairs, new_rows_size = new_rows_size_v, new_cols_size = new_cols_size_v, normalize = normalize_v, quadratic_downscale=quadratic_downscale_val)
  return x_data_fin

# call e.g. x_data, [ [(), (), ....], [(), (), ....], [(), (), ....] ] [28, 14], [14, 14], [False, False], [(False, 0) , (True, 2)]
def cantor_transform_n_units(x_data, used_pairs, new_rows_size_v, new_cols_size_v,  normalize_v, quadratic_downscale_v):
  no_of_iterations = len(used_pairs)
  i = 0
  while i < no_of_iterations:
    x_data = cantor_transformation_full_flow(x_data, used_pairs[i], new_rows_size = new_rows_size_v[i], new_cols_size = new_cols_size_v[i], normalize = normalize_v[i], quadratic_downscale = quadratic_downscale_v[i])
    i = i + 1

  return x_data