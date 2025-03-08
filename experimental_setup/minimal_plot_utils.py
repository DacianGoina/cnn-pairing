"""
Function for plting the results, but in a more minimal way
"""
from main.data_load_utils import load_MNIST_data, import_and_split_data
from main.io_utils import read_json_file
from matplotlib import pyplot as plt
from copy import deepcopy

def plot_min_max_key(img_size, ds_type, metric):
  return img_size + ds_type + metric

# Import results from a json file, use key indexes for corresponding results
def minimal_res_plot(file_path, five_frac_key_idx = 4, eight_frac_key_idx = 16, whole_frac_key_idx = 28, res_file_path = None):
    imported_data = read_json_file(file_path)

    five_frac_key = list(imported_data.keys())[five_frac_key_idx]
    print(five_frac_key)

    eight_frac_key = list(imported_data.keys())[eight_frac_key_idx]
    print(eight_frac_key)

    whole_frac_key = list(imported_data.keys())[whole_frac_key_idx]
    print(whole_frac_key)

    epochs_no = 10
    epochs_vals = list(range(epochs_no))
    epochs_vals = [val + 1 for val in epochs_vals]

    key_vals = [five_frac_key, eight_frac_key, whole_frac_key]
    metric_vals = ['training_time', 'val_f1_score', 'val_accuracy']
    y_axis_titles = {'training_time': "Training time (s)", 'val_f1_score': "F1 score", 'val_accuracy': "Accuracy"}
    fracs_val = {five_frac_key: '0.5 frac', eight_frac_key: '0.8 frac', whole_frac_key: 'whole frac'}
    # sets_types_keys = {'28x14':'28x14', '14x14':'14x14'}

    # plot y lim (min max) used for 0.8 and 1 frac (so not for 0.5 frac)
    plot_min_max = {plot_min_max_key('28x14', "digits", "val_f1_score"): (0.9, 1),
                    plot_min_max_key('14x14', "digits", "val_f1_score"): (0.84, 1),

                    plot_min_max_key('28x14', "digits", 'val_accuracy'): (0.9, 1),
                    plot_min_max_key('14x14', "digits", 'val_accuracy'): (0.84, 1),

                    plot_min_max_key('28x14', "fashion", "val_f1_score"): (0.74, 0.9),
                    plot_min_max_key('14x14', "fashion", "val_f1_score"): (0.65, 0.9),

                    plot_min_max_key('28x14', "fashion", 'val_accuracy'): (0.74, 0.9),
                    plot_min_max_key('14x14', "fashion", 'val_accuracy'): (0.67, 0.9)
                    }

    # plot_exp_type = "digits"
    plot_exp_type = "fashion"

    counter = 0
    for key_val in key_vals:
        for metric_val in metric_vals:
            values_origin = imported_data[key_val][metric_val]['original']

            values_cantor1 = imported_data[key_val][metric_val]['cantor1']
            values_cantor2 = imported_data[key_val][metric_val]['cantor2']

            values_bicubic28x14 = imported_data[key_val][metric_val]['bicubic28x14']
            values_bicubic14x14 = imported_data[key_val][metric_val]['bicubic14x14']
            values_nearest28x14 = imported_data[key_val][metric_val]['nearest28x14']
            values_nearest14x14 = imported_data[key_val][metric_val]['nearest14x14']

            original_tag = 'Original model'
            cantor_tag = 'Proposed'
            bicubic_tag = 'Bicubic Interpolation'
            nearest_tag = 'Nearest Neighbor'

            set28x14 = [(original_tag, values_origin), (cantor_tag, values_cantor1),
                        (bicubic_tag, values_bicubic28x14), (nearest_tag, values_nearest28x14)]

            set14x14 = [(original_tag, values_origin), (cantor_tag, values_cantor2),
                        (bicubic_tag, values_bicubic14x14), (nearest_tag, values_nearest14x14)]

            sets_types = [('28x14', set28x14), ('14x14', set14x14)]

            for set_type_name, current_set in sets_types:

                plt.figure(figsize=(3.6, 2.5))

                if fracs_val[key_val] != '0.5 frac':
                    if metric_val == "val_f1_score":
                        plot_lim_values = plot_min_max[plot_min_max_key(set_type_name, plot_exp_type, metric_val)]
                        plt.ylim(plot_lim_values[0], plot_lim_values[1])

                    if metric_val == "val_accuracy":
                        plot_lim_values = plot_min_max[plot_min_max_key(set_type_name, plot_exp_type, metric_val)]
                        plt.ylim(plot_lim_values[0], plot_lim_values[1])

                for label_val, values in current_set:
                    plt.plot(epochs_vals, values, label=label_val, marker='o')

                plt_complete_title = set_type_name + "_" + y_axis_titles[metric_val] + "_" + fracs_val[key_val]
                # plt.title(plt_complete_title)
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel(y_axis_titles[metric_val], fontsize=12)
                # plt.legend(fontsize = "medium")
                plt.legend(fontsize=9)

                plt.xticks(epochs_vals, fontsize=12)
                plt.yticks(fontsize=12)
                # plt.grid(True)

                plt.grid(which='major', color='#CCCCCC', linestyle='-', linewidth=0.75)
                plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
                plt.minorticks_on()

                if res_file_path == None:
                    plt.savefig( plt_complete_title + '.png', dpi=150, bbox_inches='tight')
                else:
                    plt.savefig(res_file_path + "//" + plt_complete_title + '.png', dpi=150, bbox_inches='tight')
                plt.show()
                # plt.close()
                counter = counter + 1

    print("counter: ", counter)


# Plot results, preferable obtained from an exp that used identity configuration for feature pairing and
# one experiment that used random shuffle for feature pairing.
def plot_identity_vs_shuffle_features(file_path, res_dict_idx1 = 16,res_dict_idx2 = 18,  res_img_path = None):
    imported_data = read_json_file(file_path)

    key15 = list(imported_data.keys())[res_dict_idx1]
    print(key15)

    key18 = list(imported_data.keys())[res_dict_idx2]
    print(key18)

    epochs_no = 10
    epochs_vals = list(range(epochs_no))
    epochs_vals = [val + 1 for val in epochs_vals]

    key15_cantor1_val_f1 = imported_data[key15]['val_f1_score']['cantor1']
    key15_cantor2_val_f1 = imported_data[key15]['val_f1_score']['cantor2']

    key18_cantor1_val_f1 = imported_data[key18]['val_f1_score']['cantor1']
    key18_cantor2_val_f1 = imported_data[key18]['val_f1_score']['cantor2']

    title_to_values = [('Identity config. 28 x 14', key15_cantor1_val_f1),
                       ('Identity config. 14 x 14', key15_cantor2_val_f1),
                       ('Shuffled features 28 x 14', key18_cantor1_val_f1),
                       ('Shuffled features 14 x 14', key18_cantor2_val_f1)]

    colors_to_use = ['#f79e02', '#bf7a02', '#f2ff0f', '#b3bd02']
    color_it = 0
    plt.figure(figsize=(4, 3))
    for title_val, values in title_to_values:
        plt.plot(epochs_vals, values, label=title_val, marker='o', color=colors_to_use[color_it])
        color_it = color_it + 1

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 score', fontsize=12)
    plt.legend()
    plt.xticks(epochs_vals, fontsize=12)
    plt.yticks(fontsize=12)
    # plt.grid(True)

    plt.grid(which='major', color='#CCCCCC', linestyle='-', linewidth=0.75)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.minorticks_on()

    plt.savefig(res_img_path + '.png', dpi=150, bbox_inches='tight')
    plt.show()
