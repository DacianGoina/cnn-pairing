"""
Some tests of execution time for transformation methods (cantor, bicubic, nearest)
"""

import time
import pandas as pd
import numpy as np
from copy import deepcopy

from main.cantor_utils import generate_random_pairs, cantor_quadratic_downscaling_faster, \
    cantor_transformation_over_list_of_2d
from main.data_load_utils import load_MNIST_data
from main.transformation_utils import threeD_one_ch_to_2d, flat_to_1d, resize_list_2d_np


def exec_time_tests():
    init_size = (28, 28)
    no_of_runs = 10
    methods = ['cantor', 'bicubic-interpolation', 'nearest-neighbor']

    # load data
    X_origin, y_origin = load_MNIST_data("digits")

    exec_times = dict()
    for method_name in methods:
        exec_times[method_name] = np.zeros(no_of_runs)

    for method_name in methods:
        X = deepcopy(X_origin)
        y = deepcopy(y_origin)

        X = X.values
        X = X.reshape(-1, init_size[0], init_size[1], 1).astype('float32')
        # input: (50000, 28, 28) so start with 2d matrices not 3d (2d one ch)
        X = threeD_one_ch_to_2d(X)

        # print(type(X))
        # print(X.shape)

        start_time = None
        end_time = None

        new_size = (28, 14)

        for it_no in range(no_of_runs):

            if method_name == 'cantor':

                start_time = time.time()
                pairs = list(range(init_size[0] * init_size[1]))
                pairs = generate_random_pairs(pairs, shuffle_seed=0)
                X = flat_to_1d(X)  # flatten (to 1d)
                input_x_1d_after_mapping = cantor_transformation_over_list_of_2d(X, pairs)
                input_x_1d_after_mapping_tr = input_x_1d_after_mapping.transpose()

                back_to_2d = input_x_1d_after_mapping_tr.reshape(input_x_1d_after_mapping_tr.shape[0], new_size[0],
                                                                 new_size[1])
                back_to_2d = cantor_quadratic_downscaling_faster(back_to_2d, with_floor=False)
                max_value = np.max(back_to_2d)
                back_to_2d = back_to_2d / max_value
                # X = cantor_transform_one_unit(X, pairs, new_size[0], new_size[1], normalize_v = True)
                end_time = time.time()

            elif method_name == 'bicubic-interpolation':

                start_time = time.time()
                X = resize_list_2d_np(X, new_size, 'bicubic')
                end_time = time.time()

            else:
                start_time = time.time()
                X = resize_list_2d_np(X, new_size, 'nearest')
                end_time = time.time()

            print(method_name, ": %s seconds ---" % (end_time - start_time))
            exec_times[method_name][it_no] = end_time - start_time

    # print mean and std for the results
    for key_val, data_val in exec_times.items():
        print("method: ", key_val)
        df_for_res = pd.DataFrame(exec_times[key_val])
        print(type(df_for_res.describe()))
        print(df_for_res.describe())
        print("-" * 15)

