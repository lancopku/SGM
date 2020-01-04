import os
import codecs
import numpy as np
from sklearn import metrics


def eval_metrics(reference, candidate, label_dict, log_path):

    ref_dir = log_path + 'reference/'
    cand_dir = log_path + 'candidate/'
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)
    if not os.path.exists(cand_dir):
        os.makedirs(cand_dir)
    ref_file = ref_dir + 'reference'
    cand_file = cand_dir + 'candidate'

    for i in range(len(reference)):
        with codecs.open(ref_file+str(i), 'w', 'utf-8') as f:
            f.write(" ".join(reference[i]) + '\n')
        with codecs.open(cand_file+str(i), 'w', 'utf-8') as f:
            f.write(" ".join(candidate[i]) + '\n')
        
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
        micro_f1 = metrics.f1_score(y, y_pre, average='micro')
        micro_precision = metrics.precision_score(y, y_pre, average='micro')
        micro_recall = metrics.recall_score(y, y_pre, average='micro')
        instance_f1 = metrics.f1_score(y, y_pre, average='samples')
        instance_precision = metrics.precision_score(y, y_pre, average='samples')
        instance_recall = metrics.recall_score(y, y_pre, average='samples')
        return hamming_loss, \
               micro_f1, micro_precision, micro_recall, \
               instance_f1, instance_precision, instance_recall

    y, y_pre = prepare_label(reference, candidate, label_dict)
    hamming_loss, micro_f1, micro_precision, micro_recall, instance_f1, instance_precision, instance_recall = get_metrics(y, y_pre)
    
    return {'hamming_loss': hamming_loss, 
            'micro_f1': micro_f1,
            'micro_precision': micro_precision, 
            'micro_recall': micro_recall,
            'instance_f1': instance_f1,
            'instance_precision': instance_precision,
            'instance_recall': instance_recall}