import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import os
import csv

DEFAULT_DATA_PATH = 'riri145/img/'
DEFAULT_CSV_PATH = 'riri145/clean-data.csv'
DEFAULT_SAVED_LABELS = 'riri145/preloaded.pt'

class InstagramDataset(Dataset):
    '''
    Characterizes a dataset for PyTorch.
    '''
    def __init__(self, dataset_path=DEFAULT_DATA_PATH, csv_path=DEFAULT_CSV_PATH,
                label_path = DEFAULT_SAVED_LABELS, transform=None):

        # Checks if pre-saved training labels are available.
        # If not, loads from csv file and saves to a .pt file
        # (pytorch default save extension) to be loaded up in the future.
        if not os.path.exists(label_path):

            # Opens CSV file for reading
            csv_file = open(csv_path)
            csv_reader = csv.reader(csv_file, delimiter=',')

            # Creates dictionary to save all image names and labels
            data_dict = {
                'image_names': [],
                'labels': [],
            }

            # Iterates through csv file and grabs image names + labels
            for idx, line in enumerate(csv_reader):
                if idx > 0 and ('jpg' in line[3] or 'png' in line[3]):
                    data_dict['image_names'].append(line[3])
                    if int(line[-1]) == 1:
                        data_dict['labels'].append(torch.tensor([1, 0], dtype=torch.float32))
                    else:
                        data_dict['labels'].append(torch.tensor([0, 1], dtype=torch.float32))

            # Saves for easy loading next time
            if not os.path.isdir(label_path[:label_path.rfind('/')]):
                os.makedirs(label_path[:label_path.rfind('/')])
            torch.save(data_dict, label_path)

        # Otherwise, just load the pre-saved dict.
        else:
            data_dict = torch.load(label_path)

        # Saves state variables
        self.data_dict = data_dict
        self.dataset_path = dataset_path
        self.label_path = label_path
        self.transform = transform


    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.data_dict['labels'])


    def __getitem__(self, index):
        '''Generates one sample of data'''
        # Select sample
        image_name = self.data_dict['image_names'][index]

        # Load data and get label (hacky - already preprocessed)
        X = self.transform(Image.open(self.dataset_path + image_name))
        y = self.data_dict['labels'][index]

        return X, y


def get_dataloaders(dataset_path=DEFAULT_DATA_PATH, csv_path=DEFAULT_CSV_PATH,
                    label_path = DEFAULT_SAVED_LABELS, val_split=0.2, batch_sz=4,
                    num_threads=1, shuffle_val=True):
    '''
    Grabs dataloaders for train/val sets.
    
    Keyword arguments:
    > dataset_path (string) -- Path to folder where all dataset images are stored.
    > csv_path (string) -- Path to csv file with image names and labels.
    > label_path (string) -- Path to saved labels (should be .pt file).
    > val_split (float) -- Fraction of training data to be used as validation set.
    > batch_sz (int) -- Batch size to be grabbed from DataLoader.
    > num_threads (int) -- Number of threads with which to load data.
    > shuffle_val (bool) -- Whether to shuffle validation set indices.

    Return value: (train_dataloader, test_dataloader)
    > train_dataloader -- a torch.utils.data.DataLoader wrapper around
        the specified dataset's training set.
    > val_dataloader -- a torch.utils.data.DataLoader wrapper around
        the specified dataset's validation set.
    '''

    # Describes the transforms we want. Using randomCrop and toTensor.
    transform = transforms.Compose([
            transforms.Resize((128, 128)), # 128 x 128 random crop of image.
            transforms.ToTensor(),
        ])

    # Constructs InstagramDataset to load data from
    dataset = InstagramDataset(dataset_path=dataset_path, csv_path=csv_path,
                                label_path=label_path, transform=transform)

    # Grabs train/val split
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))

    # Shuffle indices if ncessary for slicing val set
    if shuffle_val:
        np.random.shuffle(indices)

    # Performs train/val split
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)

    # Constructs dataloader wrappers around InstagramDataset training and test sets
    train_dataloader = DataLoader(dataset, batch_size=batch_sz, 
                                  num_workers=num_threads, sampler=train_sampler)
    val_dataloader = DataLoader(dataset, batch_size=batch_sz, 
                                num_workers=num_threads, sampler=val_sampler)

    return (train_dataloader, val_dataloader)


def main():
    train_dataloader, val_dataloader = get_dataloaders()
    for i, thing in enumerate(train_dataloader):
        input, output = thing
        print(input.shape, output, output.shape)
        if i == 5: break


if __name__ == '__main__':
    main()
