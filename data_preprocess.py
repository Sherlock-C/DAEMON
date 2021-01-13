import ast
import csv
import os
from pickle import dump
import xlrd
import numpy as np

#
output_folder = 'datasets'
if not os.path.exists(os.path.join(os.getcwd(),output_folder)):
    os.makedirs(os.path.join(os.getcwd(),output_folder))

if not os.path.exists(os.path.join(os.getcwd(), output_folder, 'train')):
    os.makedirs(os.path.join(os.getcwd(), output_folder, 'train'))

if not os.path.exists(os.path.join(os.getcwd(), output_folder, 'test')):
    os.makedirs(os.path.join(os.getcwd(), output_folder, 'test'))

if not os.path.exists(os.path.join(os.getcwd(), output_folder, 'test_label')):
    os.makedirs(os.path.join(os.getcwd(), output_folder, 'test_label'))


def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(os.getcwd(), dataset_folder, category, filename),
                         dtype=np.float32,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)

    if not os.path.exists(os.path.join(os.getcwd(), output_folder, category, dataset)):
        os.makedirs(os.path.join(os.getcwd(), output_folder, category, dataset))

    filename = filename.strip('.txt')
    with open(os.path.join(os.getcwd(), output_folder, category, dataset, filename + ".pkl"), "wb") as file:
        dump(temp, file)


def load_data(dataset):
    if dataset == 'SMD':
        dataset_folder = 'ServerMachineDataset'
        file_list = os.listdir(os.path.join(os.getcwd(), dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith('.txt'):
                load_and_save('train', filename, dataset, dataset_folder)
                load_and_save('test', filename, dataset, dataset_folder)
                load_and_save('test_label', filename, dataset, dataset_folder)
    elif dataset == 'SMAP' or dataset == 'MSL':
        dataset_folder = os.path.join(os.getcwd(), 'data')
        with open(os.path.join(dataset_folder, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])

        data_info = [row for row in res if row[1] == dataset and row[0] != 'P-2']

        for row in data_info:

            anomalies = ast.literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool)
            filename = row[0]

            dir_type = ["train", "test", "test_label"]
            for i in dir_type:
                if not os.path.exists(os.path.join(os.getcwd(), output_folder, i, dataset)):
                    os.makedirs(os.path.join(os.getcwd(), output_folder, i, dataset))

            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = 1.0

            print(filename, 'test_label', len(label))
            with open(os.path.join(os.getcwd(), output_folder, "test_label", dataset, filename + ".pkl"), "wb") as file:
                dump(label, file)

            temp = np.load(os.path.join(dataset_folder, "train", filename + '.npy'))

            with open(os.path.join(os.getcwd(), output_folder, "train", dataset, filename + ".pkl"), "wb") as file:
                dump(temp, file)

            temp = np.load(os.path.join(dataset_folder, "test", filename + '.npy'))

            with open(os.path.join(os.getcwd(), output_folder, "test", dataset, filename + ".pkl"), "wb") as file:
                dump(temp, file)


    elif dataset == 'SWAT':
        train_table = xlrd.open_workbook(os.path.join(os.getcwd(), 'Swat', 'Physical',
                                                      'SWaT_Dataset_Normal_v0.xlsx')).sheets()[0]

        train_row = train_table.nrows

        train = []

        for i in range(2, train_row):

            train.append(train_table.row_values(i))

        train = np.asarray(train)

        m, n = train.shape
        print(m, n)

        train_samples = train[21600:, 1:n-1]

        if not os.path.exists(os.path.join(os.getcwd(), output_folder, "train", dataset)):
            os.makedirs(os.path.join(os.getcwd(), output_folder, "train", dataset))

        with open(os.path.join(os.getcwd(), output_folder, "train", dataset, "swat.pkl"), "wb") as file:
            dump(train_samples, file)

        test_table = xlrd.open_workbook(os.path.join(os.getcwd(), 'Swat', 'Physical',
                                                     'SWaT_Dataset_Attack_v0.xlsx')).sheets()[0]

        test_row = test_table.nrows

        test_data = []

        for i in range(2, test_row):
            test_data.append(test_table.row_values(i))

        test_data = np.asarray(test_data)

        test_data = test_data[:, 1:]  #

        m, n = test_data.shape
        print(m, n)

        test_samples = test_data[:, 0:n - 1]
        label = test_data[:, n-1]

        for i in range(len(label)):
            if label[i] == 'Normal':
                label[i] = 0.0
            else:
                label[i] = 1.0

        if not os.path.exists(os.path.join(os.getcwd(), output_folder, "test", dataset)):
            os.makedirs(os.path.join(os.getcwd(), output_folder, "test", dataset))

        with open(os.path.join(os.getcwd(), output_folder, "test", dataset, "swat.pkl"), "wb") as file:
            dump(test_samples, file)

        if not os.path.exists(os.path.join(os.getcwd(), output_folder, "test_label", dataset)):
            os.makedirs(os.path.join(os.getcwd(), output_folder, "test_label", dataset))

        with open(os.path.join(os.getcwd(), output_folder, "test_label", dataset, "swat.pkl"), "wb") as file:
            dump(label, file)


if __name__ == '__main__':

    datasets = ['SMD', 'SMAP', 'MSL', 'SWAT']

    for d in datasets:
            load_data(d)
