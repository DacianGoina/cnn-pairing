from io_utils import read_json_file
from matplotlib import pyplot as plt

print("Main")

res = read_json_file("../exported_data/FASHION-analytic-results.json")
print("Main")

keys = ['used_data_percentage_val = 0.5;split_seeds_val = 11;split_percentage_test_val = 0.2;split2_seeds_val = 77;seeds = [0, 0]_0',
        'used_data_percentage_val = 0.8;split_seeds_val = 11;split_percentage_test_val = 0.2;split2_seeds_val = 77;seeds = [0, 0]_12',
        'used_data_percentage_val = 1;split_seeds_val = 11;split_percentage_test_val = 0.2;split2_seeds_val = 77;seeds = [0, 0]_24']

epochs = list(range(10))

for key_v in keys:
    y_values = res[key_v]['training_time']['cantor1']
    plt.plot(epochs, y_values, label='Training_time_data')

    # Add labels and a title
    plt.xlabel('epoch')
    plt.ylabel('training time values')
    plt.title('Training time data')
    plt.show()