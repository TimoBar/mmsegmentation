import os
import re
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--work-dir', default='.')
    args = parser.parse_args()
    print(f'{args=}')
    read_training_time(args.work_dir)

def get_log_file_dirs(root_dir):
    filenames = os.listdir(root_dir)

    result = []
    for filename in filenames:
        if os.path.isdir(os.path.join(root_dir, filename)):
            result.append(os.path.join(root_dir, filename))

    result.sort()
    return result

def get_log_files(log_file_dirs):
    r = []
    for dir in log_file_dirs:
        filenames = os.listdir(dir)
        for filename in filenames:
            if not os.path.isdir(os.path.join(dir, filename)):
                r.append(os.path.join(dir, filename))

    r.sort(reverse=False)
    return r

def get_train_iter_lines(log_files):
    d = {}
    for log_file in log_files:
        log_file = open(log_file, 'r')
        lines = log_file.readlines()
        for line in lines:
            x = re.search(
                "Iter\(train\) \[[ ]*[0-9]*/[0-9]*]  base_lr: [0-9e.\+-]* lr: [0-9e.\+-]*  eta: [0-9e.\-+ a-z]*[,]?[0-9e.\-+ a-z:]* time: [0-9e.\-+]*  data_time: [0-9+e.\-]*",
                line)
            if x:
                iter_num = int(x[0].split("[")[1].split("/")[0].replace(" ", ""))
                d[iter_num] = x[0]

    return d
def read_training_time(work_dir):

    dirs = get_log_file_dirs(work_dir)
    log_files = get_log_files(dirs)

    iter_lines = get_train_iter_lines(log_files)

    time_sum = 0
    data_time_sum = 0
    for v in iter_lines.values():
        while "  " in v:
            v = v.replace("  ", " ")
        time = float(v[v.find("time:"):].split(" ")[1])
        time_sum += time
        data_time = float(v[v.find("data_time:"):].split(" ")[1])
        data_time_sum += data_time
    print(time_sum / 60)
    print(data_time_sum / 60)


if __name__ == '__main__':
    main()