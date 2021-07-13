import sys
import torch
import numpy as np
from daemon.data_utils import *
from daemon.options import Options
from daemon.model import *
from torch.utils.data import DataLoader
import os


opt = Options().parse()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed_value):
    if seed_value == -1:
        return

    import random

    random.seed(seed_value)

    torch.manual_seed(seed_value)

    torch.cuda.manual_seed(seed_value)

    torch.cuda.manual_seed_all(seed_value)

    np.random.seed(seed_value)

    torch.backends.cudnn.deterministic = True

seed = 12345

set_seed(seed)

batchsize = 50

opt.train_batchsize = batchsize
opt.val_batchsize = batchsize
opt.test_batchsize = batchsize

opt.niter = 20

kernel_size = 32

opt.ngf = kernel_size
opt.ndf = kernel_size

def anomaly_detection(data_type):

    with open('./detection_results_' + str(data_type) +'.txt', 'w') as f:
        f.write('channel' + '\t' + 'dataset' + '\t' + 'f1' + '\t' + 'pre' + '\t' + 'rec' + '\t' +
                'tp' + '\t' + 'tn' + '\t' + 'fp' + '\t' + 'fn' + '\t' + 'train_time' + '\t' + 'epoch_time' +
                 '\t' +'test_time' + '\t' + 'latency' + '\n')

        total_tp = 0.0
        total_tn = 0.0
        total_fp = 0.0
        total_fn = 0.0
        total_latency = 0.0
        total_train_time = 0.0
        total_test_time = 0.0
        total_epoch_time = 0.0
        opt.dataset = data_type

        if data_type == 'SMAP':
            opt.dim = 25
            opt.w_lat = 1
            opt.w_rec = 0.1

        elif data_type == 'MSL':
            opt.dim = 55
            opt.w_lat = 1
            opt.w_rec = 1

        elif data_type == 'SMD':
            opt.dim = 38
            opt.w_lat = 1
            opt.w_rec = 1

        elif data_type == 'SWAT':
            opt.dim = 51
            opt.w_lat = 1.0
            opt.w_rec = 1.0
            opt.step = 10

        path_train = os.path.join(os.getcwd(), "datasets", "train", data_type)
        files = os.listdir(path_train)

        for file in files:
            opt.filename = file
            seed = 12345
            set_seed(seed)
            data_name = data_type + '/' + str(file)
            print('file=', data_name)
            samples_train_data, samples_val_data = read_train_data(opt.window_size, file=data_name,
                                                                   step=opt.step)
            print('train samples', samples_train_data.shape)
            train_data = DataLoader(dataset=samples_train_data, batch_size=opt.train_batchsize, shuffle=True)
            val_data = DataLoader(dataset=samples_val_data, batch_size=opt.val_batchsize, shuffle=True)

            samples_test_data, test_label = read_test_data(opt.window_size, file=data_name)

            test_data = DataLoader(dataset=samples_test_data, batch_size=opt.test_batchsize)

            model = DAEMON(opt, train_data, val_data, test_data, test_label, device)

            train_time, epoch_time = model.train()

            model.load()

            f1, pre, rec, tp, tn, fp, fn, latency, test_time = model.eval_result(test_data, test_label)

            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_latency += latency
            total_pre = total_tp / (total_tp + total_fp)
            total_rec = total_tp / (total_tp + total_fn)
            total_f1 = 2*total_pre*total_rec / (total_pre + total_rec)
            total_train_time += train_time
            total_test_time += test_time
            total_epoch_time += epoch_time

            print(str(data_type) + '\t' + str(file) + '\tf1=' + str(f1) + '\tpre=' + str(pre) +
                  '\trec=' + str(rec) + '\ttp=' + str(tp) + '\ttn=' + str(tn) + '\tfp=' + str(fp) +
                  '\tfn=' + str(fn) + '\tlatency=' + str(latency))

            print('total results:' + '\tt_f1=' + str(total_f1) + '\tt_pre=' + str(total_pre) +
                  '\tt_rec=' + str(total_rec) + '\tt_tp=' + str(total_tp) + '\tt_tn=' + str(total_tn) +
                  '\tt_fp=' + str(total_fp) + '\tt_fn=' + str(total_fn) + '\tt_latency=' + str(total_latency)
                  + '\tt_epoch_time=' + str(total_epoch_time))

            f.write(str(data_type) + '\t' + str(file) + '\t' + str(f1) + '\t' + str(pre) + '\t' +
                    str(rec) + '\t' + str(tp) + '\t' + str(tn) + '\t' + str(fp) + '\t' + str(fn) +
                    '\t' + str(train_time) + '\t' + str(epoch_time) + '\t' + str(test_time) +
                    '\t' + str(latency) + '\n')


    print('finished')


if __name__ == '__main__':

    commands = sys.argv[1]

    anomaly_detection(commands)
