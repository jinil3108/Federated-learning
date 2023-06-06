# Setting the clients, and another values regarding the federated learning.

import torch
import warnings

# Number of clients.
num_clients = 5
num_selected = 2           # Usually it is 30% of total clients.
num_rounds = 150            # In each round, number of clients are randomly selected.
epochs = 5                  # For each client, total number of training round.
batch_size = 32             # Loading the data into data loader by batches.
total_covid_images = 1252
total_non_covid_images = 1230

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "Dataset"
source_dir = ["non-COVID", "COVID"]


if __name__ == '__main__':
    # Setting the seed for generating the random numbers.
    torch.manual_seed(0)

    # Checking whether the pytorch is working on the system or not.
    print('Using PyTorch version: ', torch.__version__)

    # Setting the device accordingly to the cude available in the system or not.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print('Default GPU Device: {}'.format(torch.cuda.get_device_name(0)))
    else:
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
