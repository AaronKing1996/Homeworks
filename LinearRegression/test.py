import numpy as np
import pandas as pd
import csv


def test(csv_file, output_file):
    with open(csv_file, 'r', encoding='big5') as csv_file:
        full_data = pd.read_csv(csv_file, header=None)
        full_data[full_data == 'NR'] = 0

        test_data = full_data.iloc[:, 2:]
        test_data = test_data.to_numpy()
        test_x = np.empty([240, 18 * 9], dtype=float)
        # reshape成一行
        for i in range(240):
            test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)

        # normalize
        print("normalize...")
        # 压缩行，求列的平均值
        mean_x = np.mean(test_x, axis=0)
        std_x = np.std(test_x, axis=0)

        for i in range(len(test_x)):
            for j in range(len(test_x[0])):
                if std_x[j] != 0:
                    test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
        test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)

        # test
        w = np.load(output_file + 'weight.npy')
        ans_y = np.dot(test_x, w)

        # save
        with open(output_file + 'submit.csv', mode='w', newline='') as submit_file:
            csv_writer = csv.writer(submit_file)
            header = ['id', 'value']
            print(header)
            csv_writer.writerow(header)
            for i in range(240):
                row = ['id_' + str(i), ans_y[i][0]]
                csv_writer.writerow(row)
                print(row)
