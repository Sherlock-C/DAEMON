import numpy as np

from sklearn.preprocessing import MinMaxScaler



def average_filter(values, n=3):
    """
    Calculate the sliding window average for the give time series.
    Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
    :param values: list.
        a list of float numbers
    :param n: int, default 3.
        window size.
    :return res: list.
        a list of value after the average_filter process.
    """

    if n >= len(values):
        n = len(values)

    res = np.cumsum(values, dtype=float)
    res[n:] = res[n:] - res[:-n]
    res[n:] = res[n:] / n

    for i in range(1, n):
        res[i] /= (i + 1)

    return res

def spectral_residual_transform(values):
    """
    This method transform a time series into spectral residual series
    :param values: list.
        a list of float values.
    :return: mag: list.
        a list of float values as the spectral residual values
    """

    EPS = 1e-8

    trans = np.fft.fft(values)
    mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)
    eps_index = np.where(mag <= EPS)[0]
    mag[eps_index] = EPS

    mag_log = np.log(mag)
    mag_log[eps_index] = 0

    spectral = np.exp(mag_log - average_filter(mag_log, n=3))

    trans.real = trans.real * spectral / mag
    trans.imag = trans.imag * spectral / mag
    trans.real[eps_index] = 0
    trans.imag[eps_index] = 0

    wave_r = np.fft.ifft(trans)
    mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)
    return mag


def proprocess(df):
    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError("Data must be 2-D array")

    if np.any(sum(np.isnan(df)) != 0):
        print("Data contains nan. Will be repalced with 0")

        df = np.nan_to_num()

    df = MinMaxScaler().fit_transform(df)

    print("Data is normalized [0,1]")

    return df


def read_train_data(seq_length, file = '', step=1, valid_portition=0.3):

    values = []

    df = np.load('./datasets/train/' + file, allow_pickle=True)
    print(df.shape)

    (whole_len, whole_dim) = df.shape

    for i in range(whole_dim):
        df[:, i] = spectral_residual_transform(df[:, i])

    print('SR')

    values = proprocess(df)


    n = int(len(values) * valid_portition)

    if n > seq_length:#the length of validation set must be larger than the length of window size
        train, val = values[:-n], values[-n:]

        if step == 1:
            num_samples_train = (train.shape[0] - seq_length) + 1
            num_samples_val = (val.shape[0] - seq_length) + 1

        else:
            num_samples_train = (train.shape[0] - seq_length) // step
            num_samples_val = (val.shape[0] - seq_length) // step

        temp_train = np.empty([num_samples_train, seq_length, whole_dim])

        temp_val = np.empty([num_samples_val, seq_length, whole_dim])

        for i in range(num_samples_train):
            for j in range(train.shape[1]):
                temp_train[i, :, j] = train[(i*step):(i*step + seq_length), j]

        for i in range(num_samples_val):
            for j in range(val.shape[1]):
                temp_val[i, :, j] = val[(i*step):(i*step + seq_length), j]

        train_data = temp_train

        val_data = temp_val

    else:
        train = values

        if step == 1:
            num_samples_train = (train.shape[0] - seq_length) + 1


        else:
            num_samples_train = (train.shape[0] - seq_length) // step


        temp_train = np.empty([num_samples_train, seq_length, whole_dim])


        for i in range(num_samples_train):
            for j in range(train.shape[1]):
                temp_train[i, :, j] = train[(i * step):(i * step + seq_length), j]

        train_data = temp_train

        val_data = train_data

    return train_data, val_data

def read_test_data(seq_length, file = ''):

    df = np.load('./datasets/test/' + file, allow_pickle=True)
    label = np.load('./datasets/test_label/' + file, allow_pickle=True).astype(np.float)
    print(df.shape, label.shape)

    (whole_len, whole_dim) = df.shape

    for i in range(whole_dim):
        df[:, i] = spectral_residual_transform(df[:, i])

    print('SR')

    test = proprocess(df)



    num_samples_test = (test.shape[0] - seq_length) + 1

    temp_test = np.empty([num_samples_test, seq_length, whole_dim])


    for i in range(num_samples_test):
        for j in range(test.shape[1]):
                temp_test[i, :, j] = test[(i):(i + seq_length), j]


    test_data = temp_test

    return test_data, label