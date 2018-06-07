import os
import csv
import codecs
import yaml
import time
import numpy as np

from sklearn import metrics


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):

    return AttrDict(yaml.load(open(path, 'r')))


def read_datas(filename, trans_to_num=False):
    lines = open(filename, 'r').readlines()
    lines = list(map(lambda x: x.split(), lines))
    if trans_to_num:
        lines = [list(map(int, line)) for line in lines]
    return lines


def save_datas(data, filename, trans_to_str=False):
    if trans_to_str:
        data = [list(map(str, line)) for line in data]
    lines = list(map(lambda x: " ".join(x), data))
    with open(filename, 'w') as f:
        f.write("\n".join(lines))


def logging(file):
    def write_log(s):
        print(s, end='')
        with open(file, 'a') as f:
            f.write(s)
    return write_log


def logging_csv(file):
    def write_csv(s):
        with open(file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(s)
    return write_csv


def format_time(t):
    return time.strftime("%Y-%m-%d-%H:%M:%S", t)


def eval_metrics(reference, candidate, label_dict, log_path):
    ref_dir = log_path + 'reference/'
    cand_dir = log_path + 'candidate/'
    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    if not os.path.exists(cand_dir):
        os.mkdir(cand_dir)
    ref_file = ref_dir+'reference'
    cand_file = cand_dir+'candidate'

    for i in range(len(reference)):
        with codecs.open(ref_file+str(i), 'w', 'utf-8') as f:
            f.write("".join(reference[i])+'\n')
        with codecs.open(cand_file+str(i), 'w', 'utf-8') as f:
            f.write("".join(candidate[i])+'\n')
        
    def make_label(l, label_dict):
        length = len(label_dict)
        result = np.zeros(length)
        indices = [label_dict.get(label.strip().lower(), 0) for label in l]
        result[indices] = 1
        return result

    def prepare_label(y_list, y_pre_list, label_dict):
        reference = np.array([make_label(y, label_dict) for y in y_list])
        candidate = np.array([make_label(y_pre, label_dict) for y_pre in y_pre_list])
        return reference, candidate

    def get_metrics(y, y_pre):
        hamming_loss = metrics.hamming_loss(y, y_pre)
        macro_f1 = metrics.f1_score(y, y_pre, average='macro')
        macro_precision = metrics.precision_score(y, y_pre, average='macro')
        macro_recall = metrics.recall_score(y, y_pre, average='macro')
        micro_f1 = metrics.f1_score(y, y_pre, average='micro')
        micro_precision = metrics.precision_score(y, y_pre, average='micro')
        micro_recall = metrics.recall_score(y, y_pre, average='micro')
        return hamming_loss, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall


    y, y_pre = prepare_label(reference, candidate, label_dict)
    hamming_loss, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall = get_metrics(y, y_pre)
    return {'hamming_loss': hamming_loss, 
            'macro_f1': macro_f1,
            'macro_precision': macro_precision, 
            'macro_recall': macro_recall,
            'micro_f1': micro_f1,
            'micro_precision': micro_precision, 
            'micro_recall': micro_recall}
