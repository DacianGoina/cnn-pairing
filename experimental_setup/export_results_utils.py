"""
File with utils for importing the experiment results (from json files)
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy

from main.cantor_utils import cantor_quadratic_downscaling_faster, cantor_transform_one_unit, generate_random_pairs
from main.data_load_utils import import_and_split_data, load_MNIST_data
from main.io_utils import export_dict_as_json
from main.transformation_utils import resize_list_2d_np, threeD_one_ch_to_2d


# obs: pair seed none for cantor0 (no transformation)
# e.g. of usage:
#   cantor1 = read_json_file("MNIST-FASHION-CANTOR-ONE-TRANSFORMATION.json")
#   get_test_by_params(cantor1,  used_data_percentage_val, split_seeds_val, split_percentage_test_val, split2_seeds_val, pair_seed = [ pairs_seeds_val[0] ])
def get_test_by_params(dict_obj, used_data_percentage_val = 0, split_seeds_val = 0, split_percentage_test_val = 0, split2_seeds_val = 0, pair_seed = None) :
  l = [item for key_val, item in dict_obj.items() if "TEST" in  key_val]
  s_item = None
  if pair_seed is None:
      s_item = [item for item in l if item['used_data_percentage_val'] == used_data_percentage_val
            and item['split_seeds_val'] == split_seeds_val
            and item['split_percentage_test_val'] == split_percentage_test_val
            and item['split2_seeds_val'] == split2_seeds_val]
  elif len(pair_seed) == 1:
    s_item = [item for item in l if item['used_data_percentage_val'] == used_data_percentage_val
              and item['split_seeds_val'] == split_seeds_val
              and item['split_percentage_test_val'] == split_percentage_test_val
              and item['split2_seeds_val'] == split2_seeds_val
              and item['seed_to_pairs_val'][0] ==  pair_seed[0] ]

  # len pair_seed = 2
  else:
    s_item = [item for item in l if item['used_data_percentage_val'] == used_data_percentage_val
              and item['split_seeds_val'] == split_seeds_val
              and item['split_percentage_test_val'] == split_percentage_test_val
              and item['split2_seeds_val'] == split2_seeds_val
              and item['seed_to_pairs_val'][0][0] ==  pair_seed[0] and item['seed_to_pairs_val'][1][0] ==  pair_seed[1]]

  s_item = s_item[0]
  return s_item


# Export results extracted with get_test_by_params() as plots
# it will be an image with a plot for each metric recorded from model training (acc, val_acc, f1, val_f1) etc
# params cantor1, cantor2 ... represents the results imported from json files using get_test_by_params() function
# res_img_path is the saving path for the created image, if None then save in local directory using a created title (from seeds value etc)

def export_results_as_plots(cantor1, cantor2, original, bicubic28x14, bicubic14x14, nearest28x14, nearest14x14, res_img_path = None):
    used_data_percentage_vals = [0.5, 0.8, 1]
    split_seeds_vals = [11, 104]
    split_percentage_test_vals = [0.2]
    split2_seeds_vals = [77, 12]

    seeds_values = [[0, 0], [0, 5], [5, 11]]

    all_combinations = list(
        itertools.product(used_data_percentage_vals, split_seeds_vals, split_percentage_test_vals, split2_seeds_vals,
                          seeds_values))

    it_val = 0

    for used_data_percentage_val, split_seeds_val, split_percentage_test_val, split2_seeds_val, seeds_values_val in all_combinations:

        cantor1_d = get_test_by_params(cantor1, used_data_percentage_val, split_seeds_val, split_percentage_test_val,
                                       split2_seeds_val, pair_seed=[seeds_values_val[0]])
        cantor2_d = get_test_by_params(cantor2, used_data_percentage_val, split_seeds_val, split_percentage_test_val,
                                       split2_seeds_val, pair_seed=seeds_values_val)
        original_d = get_test_by_params(original, used_data_percentage_val, split_seeds_val, split_percentage_test_val,
                                        split2_seeds_val, pair_seed=None)

        bicubic28x14_d = get_test_by_params(bicubic28x14, used_data_percentage_val, split_seeds_val,
                                            split_percentage_test_val, split2_seeds_val, pair_seed=None)
        bicubic14x14_d = get_test_by_params(bicubic14x14, used_data_percentage_val, split_seeds_val,
                                            split_percentage_test_val, split2_seeds_val, pair_seed=None)
        nearest28x14_d = get_test_by_params(nearest28x14, used_data_percentage_val, split_seeds_val,
                                            split_percentage_test_val, split2_seeds_val, pair_seed=None)
        nearest14x14_d = get_test_by_params(nearest14x14, used_data_percentage_val, split_seeds_val,
                                            split_percentage_test_val, split2_seeds_val, pair_seed=None)

        it_no = 10
        metrics_name_it = 0

        # maybe lets use a order for them
        metrics_names = ['accuracy', 'val_accuracy', 'loss', 'val_loss', 'precision', 'val_precision', 'recall',
                         'val_recall', 'f1_score', 'val_f1_score', 'auc', 'val_auc', ]

        # Create the grid
        rows, cols = 7, 2

        # Set up the figure and subplots
        fig, axes = plt.subplots(rows, cols, figsize=(15, 27))
        epochs_vals = np.array(list(range(it_no))) + 1

        # for all metrics
        for i in range(1, rows):
            for j in range(cols):
                # plot values for current metric
                current_metric_name = metrics_names[metrics_name_it]

                axes[i, j].plot(epochs_vals, cantor1_d['metrics_res'][current_metric_name], label="cantor 1")
                axes[i, j].plot(epochs_vals, cantor2_d['metrics_res'][current_metric_name], label="cantor 2")
                axes[i, j].plot(epochs_vals, original_d['metrics_res'][current_metric_name], label="original")

                axes[i, j].plot(epochs_vals, bicubic28x14_d['metrics_res'][current_metric_name], label="bicubic28x14")
                axes[i, j].plot(epochs_vals, bicubic14x14_d['metrics_res'][current_metric_name], label="bicubic14x14")
                axes[i, j].plot(epochs_vals, nearest28x14_d['metrics_res'][current_metric_name], label="nearest28x14")
                axes[i, j].plot(epochs_vals, nearest14x14_d['metrics_res'][current_metric_name], label="nearest14x14")

                axes[i, j].set_xticks(epochs_vals)

                axes[i, j].set(title=current_metric_name.upper(), xlabel="epoch", ylabel=current_metric_name)
                axes[i, j].legend()

                metrics_name_it = metrics_name_it + 1

        # for training time
        axes[0, 0].plot(epochs_vals, cantor1_d['training_time'], label="cantor 1")
        axes[0, 0].plot(epochs_vals, cantor2_d['training_time'], label="cantor 2")
        axes[0, 0].plot(epochs_vals, original_d['training_time'], label="original")
        axes[0, 0].plot(epochs_vals, bicubic28x14_d['training_time'], label="bicubic28x14")
        axes[0, 0].plot(epochs_vals, bicubic14x14_d['training_time'], label="bicubic14x14")
        axes[0, 0].plot(epochs_vals, nearest28x14_d['training_time'], label="nearest28x14")
        axes[0, 0].plot(epochs_vals, nearest14x14_d['training_time'], label="nearest14x14")

        axes[0, 0].set_xticks(epochs_vals)

        axes[0, 0].set(title="Training times", xlabel="epoch", ylabel="training time (s)")
        axes[0, 0].legend()

        # title for whole grid

        # big_title = 'used_data_percentage_val = ' + str(used_data_percentage_val) + ", split_seeds_val = " +  \
        #              str(split_seeds_val) + ", split_percentage_test_val = " + str(split_percentage_test_val)  + ", split2_seeds_val = ", + str(split2_seeds_val)

        big_title = " ".join(
            ['used_data_percentage_val = ', str(used_data_percentage_val), ', split_seeds_val = ', str(split_seeds_val),
             ', split_percentage_test_val = ', str(split_percentage_test_val), ', split2_seeds_val = ',
             str(split2_seeds_val),
             ', seeds = ', str(seeds_values_val)])

        fig.suptitle(big_title, fontsize=16, y=1.005)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plot
        # plt.show()

        if res_img_path == None:
            img_file_name = "/content/results/" + big_title + "_" + str(it_val) + ".png"
        else:
            img_file_name = res_img_path + ".png"
        fig.savefig(img_file_name, dpi=200, bbox_inches='tight')
        plt.close()

        # just to see where we are with the plot
        it_val = it_val + 1
        print(it_val)
        # if it_val == 2:
        #   break


# Export analytic results (metric results) as json; the results are extracted from files using using get_test_by_params() function
# res_file_path is the saving path for the created json file
def export_analytic_res_as_json(cantor1, cantor2, original, bicubic28x14, bicubic14x14, nearest28x14, nearest14x14, res_file_path = None):
    used_data_percentage_vals = [0.5, 0.8, 1]
    split_seeds_vals = [11, 104]
    split_percentage_test_vals = [0.2]
    split2_seeds_vals = [77, 12]

    seeds_values = [[0, 0], [0, 5], [5, 11]]

    all_combinations = list(
        itertools.product(used_data_percentage_vals, split_seeds_vals, split_percentage_test_vals, split2_seeds_vals,
                          seeds_values))

    it_val = 0
    it_no = 10
    epochs_vals = np.array(list(range(it_no))) + 1

    final_dict = dict()
    dict_title = 'MNIST-DIGITS'
    final_dict['title'] = dict_title

    for used_data_percentage_val, split_seeds_val, split_percentage_test_val, split2_seeds_val, seeds_values_val in all_combinations:
        # for each combination - get data
        cantor1_d = get_test_by_params(cantor1, used_data_percentage_val, split_seeds_val, split_percentage_test_val,
                                       split2_seeds_val, pair_seed=[seeds_values_val[0]])
        cantor2_d = get_test_by_params(cantor2, used_data_percentage_val, split_seeds_val, split_percentage_test_val,
                                       split2_seeds_val, pair_seed=seeds_values_val)
        original_d = get_test_by_params(original, used_data_percentage_val, split_seeds_val, split_percentage_test_val,
                                        split2_seeds_val, pair_seed=None)

        bicubic28x14_d = get_test_by_params(bicubic28x14, used_data_percentage_val, split_seeds_val,
                                            split_percentage_test_val, split2_seeds_val, pair_seed=None)
        bicubic14x14_d = get_test_by_params(bicubic14x14, used_data_percentage_val, split_seeds_val,
                                            split_percentage_test_val, split2_seeds_val, pair_seed=None)
        nearest28x14_d = get_test_by_params(nearest28x14, used_data_percentage_val, split_seeds_val,
                                            split_percentage_test_val, split2_seeds_val, pair_seed=None)
        nearest14x14_d = get_test_by_params(nearest14x14, used_data_percentage_val, split_seeds_val,
                                            split_percentage_test_val, split2_seeds_val, pair_seed=None)

        # used metrics
        metrics_names = ['training_time', 'accuracy', 'val_accuracy', 'loss', 'val_loss', 'precision', 'val_precision',
                         'recall', 'val_recall', 'f1_score', 'val_f1_score', 'auc', 'val_auc', ]

        local_dict = dict()  # local dict at combination level
        local_dict['epochs_vals'] = epochs_vals.tolist()

        # for each metric (including traning time)
        for current_metric_name in metrics_names:
            local_dict_metric = dict()  # local dict with at metric level

            if current_metric_name == "training_time":
                local_dict_metric['cantor1'] = cantor1_d[current_metric_name]
                local_dict_metric['cantor2'] = cantor2_d[current_metric_name]
                local_dict_metric['original'] = original_d[current_metric_name]
                local_dict_metric['bicubic28x14'] = bicubic28x14_d[current_metric_name]
                local_dict_metric['bicubic14x14'] = bicubic14x14_d[current_metric_name]
                local_dict_metric['nearest28x14'] = nearest28x14_d[current_metric_name]
                local_dict_metric['nearest14x14'] = nearest14x14_d[current_metric_name]

            else:
                local_dict_metric['cantor1'] = cantor1_d['metrics_res'][current_metric_name]
                local_dict_metric['cantor2'] = cantor2_d['metrics_res'][current_metric_name]
                local_dict_metric['original'] = original_d['metrics_res'][current_metric_name]
                local_dict_metric['bicubic28x14'] = bicubic28x14_d['metrics_res'][current_metric_name]
                local_dict_metric['bicubic14x14'] = bicubic14x14_d['metrics_res'][current_metric_name]
                local_dict_metric['nearest28x14'] = nearest28x14_d['metrics_res'][current_metric_name]
                local_dict_metric['nearest14x14'] = nearest14x14_d['metrics_res'][current_metric_name]

            local_dict[current_metric_name] = local_dict_metric  # assign local dict metric to local dict combination

        key_name = "".join(
            ['used_data_percentage_val = ', str(used_data_percentage_val), ';split_seeds_val = ', str(split_seeds_val),
             ';split_percentage_test_val = ', str(split_percentage_test_val), ';split2_seeds_val = ',
             str(split2_seeds_val),
             ';seeds = ', str(seeds_values_val)])

        key_name = key_name + "_" + str(it_val)
        final_dict[key_name] = local_dict  # local dict at combination level

        # just to see where we are with the plot
        it_val = it_val + 1
        print(it_val)

    export_dict_as_json(final_dict, res_file_path)



# Import MNIST data with a split seed (split_seed), select a sample (by idx), created pairs for cantor pairing function (via pairs_shuffle_seed)
# For the selected sample plot it, also plot different transformed version of it (with cantor pairing, with / without quadratic downscaling)
# Also plot historgram of frequencies.
def plot_samples_and_hist(idx = 0, split_seed = 0, pairs_shuffle_seed = 15 ):
    X_origin, y_origin = load_MNIST_data("digits")

    X_no_trans = deepcopy(X_origin)
    y_no_trans = deepcopy(y_origin)

    X_for_trans = deepcopy(X_origin)
    y_for_trans = deepcopy(y_origin)

    cases = [('normal', X_no_trans, y_no_trans), ('cantor', X_for_trans, y_for_trans),
             ('bicubic', X_no_trans, y_no_trans), ('nearest', X_no_trans, y_no_trans)]

    # when you apply cantor 2 times, you need to reverse 2 times

    plot_w = 7.5
    plot_h = 3.5



    for transform_case, x_data_, y_data_ in cases:

        x_train, x_test, y_train, y_test, y_train_distribution, y_test_distribution = import_and_split_data(x_data_,
                                                                                                            y_data_,
                                                                                                            used_data_percentage=1,
                                                                                                            split_seed=22,
                                                                                                            split_percentage_test=0.1,
                                                                                                            split2_seed=45,
                                                                                                            rows_num=28,
                                                                                                            cols_num=28,
                                                                                                            no_of_classes_v=10)

        selected_sample = x_train[idx]
        selected_sample_fin = None

        if transform_case == 'cantor':
            max_cantor_possible = 130560
            sample_copy = deepcopy(selected_sample)
            x_train_transformed = cantor_transform_one_unit([selected_sample],
                                                            generate_random_pairs(list(range(28 * 28)),
                                                                                  pairs_shuffle_seed), 28, 14, False)
            x_train_transformed_normalized = cantor_transform_one_unit([sample_copy],
                                                                       generate_random_pairs(list(range(28 * 28)),
                                                                                             pairs_shuffle_seed), 28,
                                                                       14, True)
            reshaped_array_origin = x_train_transformed_normalized[0].reshape(28, 14)
            plt.imshow(reshaped_array_origin, cmap='gray')
            plt.colorbar()  # Add color bar to the plot
            plt.title("Cantor res normalized [0, 1]")  # normalized to [0, 1] directly, no squared, no floor
            plt.show()

            # scaler = MinMaxScaler(feature_range=(0, 255))
            normalized255 = deepcopy(x_train_transformed[0].flatten())
            # normalized255 = (normalized255 / max_cantor_possible) * 255
            normalized255 = ((normalized255 - np.min(normalized255)) / (
                        np.max(normalized255) - np.min(normalized255))) * 255
            bins = np.arange(256)
            hist, bin_edges = np.histogram(normalized255, bins=bins)
            plt.figure(figsize=(plot_w, plot_h))
            # Optional: Plot the histogram
            plt.bar(bin_edges[:-1], hist, width=1, align='edge', edgecolor='black', color='#23b84b')
            plt.yscale('log')
            plt.xlabel('Pixel value', fontsize=13)
            plt.ylabel('Frequency', fontsize=13)
            plt.title('cantor normalized [0, 255]')
            plt.xticks(fontsize=12)  # Increase font size for x-axis ticks
            plt.yticks(fontsize=12)
            plt.show()


            print("cantor max", np.max(x_train_transformed[0]))
            x_train_transformed_copy2 = deepcopy(x_train_transformed[0])

            vectorized_func_no_floor = np.vectorize(lambda val: (-1 + np.sqrt(1 + 8 * val)) / (2 * 2))
            selected_sample_fin_no_floor = cantor_quadratic_downscaling_faster(x_train_transformed_copy2)
            reshaped_array_no_floor = selected_sample_fin_no_floor.reshape(28, 14)
            plt.imshow(reshaped_array_no_floor / np.max(reshaped_array_no_floor), cmap='gray')
            print("no floor max", np.max(reshaped_array_no_floor))
            plt.colorbar()  # Add color bar to the plot
            plt.title("Cantor res squared")  # just squared (NO FLOOR) and normalized to [0,1 ]
            plt.show()

            vectorized_func = np.vectorize(lambda val: np.floor((-1 + np.sqrt(1 + 8 * val)) / (2 * 2)))
            selected_sample_fin = vectorized_func(x_train_transformed[0])
            reshaped_array = selected_sample_fin.reshape(28, 14)
            plt.imshow(reshaped_array, cmap='gray')
            plt.colorbar()  # Add color bar to the plot
            plt.title("Cantor res squaredf")  # just squared (with floor)
            plt.show()

            plt.imshow(reshaped_array / np.max(reshaped_array), cmap='gray')
            plt.colorbar()  # Add color bar to the plot
            plt.title("Cantor res squaredf_n")  # squared (floor) and then normalized to [0, 1]
            plt.show()

            print("max ", np.max(selected_sample_fin))
            print("min ", np.min(selected_sample_fin))
        elif transform_case == 'normal':
            selected_sample_fin = selected_sample
            plt.imshow(selected_sample_fin, cmap='gray')
            plt.colorbar()  # Add color bar to the plot
            plt.title("original")
            plt.show()

        elif transform_case == 'bicubic':
            selected_sample_fin = resize_list_2d_np([threeD_one_ch_to_2d(selected_sample)], (28, 14), 'bicubic',
                                                    normalize=False)

        elif transform_case == 'nearest':
            selected_sample_fin = resize_list_2d_np([threeD_one_ch_to_2d(selected_sample)], (28, 14), 'nearest',
                                                    normalize=False)

        selected_sample_fin = selected_sample_fin.flatten()

        bins = np.arange(256)
        hist, bin_edges = np.histogram(selected_sample_fin, bins=bins)

        plt.figure(figsize=(plot_w, plot_h))
        # Optional: Plot the histogram
        plt.bar(bin_edges[:-1], hist, width=1, align='edge', edgecolor='black', color='#23b84b')
        plt.yscale('log')
        plt.xlabel('Pixel value', fontsize=13)
        plt.ylabel('Frequency', fontsize=13)
        plt.title('case: ' + transform_case)
        plt.xticks(fontsize=12)  # Increase font size for x-axis ticks
        plt.yticks(fontsize=12)
        plt.show()

