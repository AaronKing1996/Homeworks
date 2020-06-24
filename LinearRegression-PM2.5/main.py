import train
import test
import argparse


if __name__ == "__main__":
    # param
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='./data/train.csv', help='training csv file path')
    parser.add_argument('--train_csv_skip_rows', type=int, default=0, help='skip the first `skip_rows` lines')
    parser.add_argument('--train_csv_skip_cols', type=int, default=3, help='skip the first `skip_rows` lines')
    parser.add_argument('--items_count', type=int, default=18, help='number of training items')
    parser.add_argument('--training_flag', type=int, default=0, help='training flag')

    parser.add_argument('--test_csv', type=str, default='./data/test.csv', help='test csv file path')
    parser.add_argument('--output', type=str, default='./output/', help='output file path')
    opt = parser.parse_args()
    print(opt)

    if opt.training_flag == 1:
        # train
        items, data = train.read_csv(opt.train_csv, opt.train_csv_skip_rows, opt.train_csv_skip_cols, opt.items_count)
        train.k_fold_cross_validation(data, 12, 120, opt.output)

    # test
    test.test(opt.test_csv, opt.output)
