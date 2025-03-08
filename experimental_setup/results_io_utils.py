"""
File with utils for importing the experiment results (from json files)
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np

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



