"""
Experimental setup when cantor transformation is applied 2 times (i.e. it reduces the number of features by 4 ( / 4))
"""

from main.cantor_utils import generate_pairs_instances, generate_random_pairs, cantor_transform_n_units
import itertools

from main.data_load_utils import import_and_split_data, load_MNIST_data
from main.io_utils import dict_int64_keys_to_int_keys, export_dict_as_json
from main.model_utils import create_and_train_model, get_confusion_matrix

# CELL 1 : setup
## FOR CANTOR 2 TRANSFORMATIONS

rows_num_v = 28
cols_num_v = 28

pairs_seeds = [0, 5, 11] # 0 do not shuffle
pairs_sets = generate_pairs_instances(list(range(rows_num_v * cols_num_v)), pairs_seeds)

seed_to_pairs = list(zip(pairs_seeds, pairs_sets))

seed_to_pairs_two = [
    [(pairs_seeds[0], generate_random_pairs(list(range(rows_num_v * cols_num_v)), pairs_seeds[0]) ), (pairs_seeds[0], generate_random_pairs(list(range(int((rows_num_v * cols_num_v)/2) )), pairs_seeds[0]) )],
    [(pairs_seeds[0], generate_random_pairs(list(range(rows_num_v * cols_num_v)), pairs_seeds[0]) ), (pairs_seeds[1], generate_random_pairs(list(range(int((rows_num_v * cols_num_v)/2) )), pairs_seeds[1]) )],
    [(pairs_seeds[1], generate_random_pairs(list(range(rows_num_v * cols_num_v)), pairs_seeds[1]) ), (pairs_seeds[2], generate_random_pairs(list(range(int((rows_num_v * cols_num_v)/2) )), pairs_seeds[2]) )]
]

for i in seed_to_pairs_two:
  print(i)

# used_data_percentage_vals = [0.5, 0.8, 1]
#used_data_percentage_vals = [0.1, 0.5, 0.8, 1]
#split_seeds_vals = [11, 89, 104]
#split_percentage_test_vals = [0.15, 0.2, 0.25]
# split_percentage_test_vals = [0.2, 0.25]
#split2_seeds_vals = [45, 77, 12]

used_data_percentage_vals = [0.5, 0.8, 1]
split_seeds_vals = [11, 104]
split_percentage_test_vals = [0.2]
split2_seeds_vals = [77, 12]
no_of_classes_val = 10

all_combinations = list( itertools.product(seed_to_pairs_two, used_data_percentage_vals, split_seeds_vals, split_percentage_test_vals, split2_seeds_vals) )

final_dict = dict()
final_dict['title'] = 'test_title'
final_dict['rows_num_v'] = rows_num_v
final_dict['cols_num_v'] = cols_num_v
final_dict['obs'] = [
    "The model is evaluated after each epoch on testing data, thus the results on last epoch are also the results for usage of model on testing data",
    "The training time values are in seconds",
    "The split is stratified on y values"]

print(len(all_combinations))


# load data
X, y = load_MNIST_data("digits")

# CELL 2: experiment

new_rows_size_v = [28, 14]
new_cols_size_v = [14, 14]
normalize_v = [False, True]
quadradic_downscaling_v = [(False, 0), (True, 2)]

test_title = "MNIST_FASHION_CANTOR_TWO_TRANSFORMATION"
#test_title = "MNIST_CIFAR_CANTOR_TWO_TRANSFORMATION"
it = 0

final_dict['title'] = test_title
final_dict['new_rows_size_v'] = new_rows_size_v
final_dict['new_cols_size_v'] = new_cols_size_v
final_dict['normalize_v'] = normalize_v
final_dict['quadradic_downscaling'] = quadradic_downscaling_v

for seed_to_pairs_val,  used_data_percentage_val, split_seeds_val, split_percentage_test_val, split2_seeds_val in all_combinations:
  print("it = ", str(it))
  # get split data
  x_train, x_test, y_train, y_test, y_train_distribution, y_test_distribution = import_and_split_data(X, y,
                                                           used_data_percentage = used_data_percentage_val,
                                                           split_seed = split_seeds_val,
                                                           split_percentage_test = split_percentage_test_val,
                                                           split2_seed = split2_seeds_val,
                                                           rows_num = rows_num_v, cols_num = cols_num_v, no_of_classes_v = no_of_classes_val)

  #cantor_transform_n_units(x_data, used_pairs, new_rows_size_v, new_cols_size_v,  normalize_v)

  seed_to_pairs_val_pairs_sets = [pairs_set for (seed, pairs_set) in seed_to_pairs_val]
  seed_to_pairs_val_pairs_seeds = [seed for (seed, pairs_set) in seed_to_pairs_val]
  #print(seed_to_pairs_val_pairs_seeds)
  #print(seed_to_pairs_val_pairs_sets)
  #break

  # convert data
  x_train = cantor_transform_n_units(x_train, seed_to_pairs_val_pairs_sets, new_rows_size_v, new_cols_size_v, normalize_v, quadradic_downscaling_v)
  x_test = cantor_transform_n_units(x_test, seed_to_pairs_val_pairs_sets, new_rows_size_v, new_cols_size_v, normalize_v, quadradic_downscaling_v)

  # train model, get results
  model, training_time, metrics_res = create_and_train_model(x_train, x_test, y_train, y_test, new_rows_size_v[-1], new_cols_size_v[-1])

  conf_matrix_obj = get_confusion_matrix(model, x_test, y_test, no_of_classes_val)

  local_dict = dict()
  local_dict['seed_to_pairs_val'] = seed_to_pairs_val
  local_dict['used_data_percentage_val'] = used_data_percentage_val
  local_dict['split_seeds_val'] = split_seeds_val
  local_dict['split_percentage_test_val'] = split_percentage_test_val
  local_dict['split2_seeds_val'] = split2_seeds_val
  local_dict['seed_to_pairs_val'] = seed_to_pairs_val

  local_dict['y_train_distribution'] = dict_int64_keys_to_int_keys(y_train_distribution)
  local_dict['y_test_distribution'] = dict_int64_keys_to_int_keys(y_test_distribution)

  local_dict['training_time'] = training_time
  local_dict['metrics_res'] = metrics_res
  local_dict['conf_matrix_obj'] = conf_matrix_obj.tolist()

  final_dict["TEST_" + str(it)] = local_dict

  it = it + 1

  # if it == 2:
  #   break


# x_train = cantor_transform_n_units(x_train, [seed_to_pairs[1][1]], new_rows_size_v, new_cols_size_v, normalize_v)
# x_test = cantor_transform_n_units(x_test, [seed_to_pairs[1][1]], new_rows_size_v, new_cols_size_v, normalize_v)

export_dict_as_json(final_dict,"MNIST-FASHION-CANTOR-TWO-TRANSFORMATION.json")