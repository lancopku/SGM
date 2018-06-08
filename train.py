import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data

import models
import data.dataloader as dataloader
import data.utils as utils
import data.dict as dict
from optims import Optim
import lr_scheduler as L

import os
import argparse
import time
import json
import collections
import codecs


#config
parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-config', default='config.yaml', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='', type=str,
                    help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-model', default='seq2seq', type=str,
                    help="Model selection")
parser.add_argument('-score', default='', type=str,
                    help="score_fn")
parser.add_argument('-pretrain', default=False, type=bool,
                    help="load pretrain embedding")
parser.add_argument('-notrain', default=False, type=bool,
                    help="train or not")
parser.add_argument('-limit', default=0, type=int,
                    help="data limit")
parser.add_argument('-log', default='', type=str,
                    help="log directory")
parser.add_argument('-unk', default=True, type=bool,
                    help="replace unk")
parser.add_argument('-memory', default=False, type=bool,
                    help="memory efficiency")
parser.add_argument('-label_dict_file', default='./data/data/rcv1.json', type=str,
                    help="label_dict")

opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)

# checkpoint
if opt.restore: 
    print('loading checkpoint...\n')
    checkpoints = torch.load(opt.restore)

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
#use_cuda = True
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)
print(use_cuda)

# data
print('loading data...\n')
start_time = time.time()
datas = torch.load(config.data)  
print('loading time cost: %.3f' % (time.time()-start_time))

trainset, validset = datas['train'], datas['valid']
src_vocab, tgt_vocab = datas['dicts']['src'], datas['dicts']['tgt']
config.src_vocab = src_vocab.size()
config.tgt_vocab = tgt_vocab.size()

trainloader = dataloader.get_loader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=2)
validloader = dataloader.get_loader(validset, batch_size=config.batch_size, shuffle=False, num_workers=2)

if opt.pretrain:
    pretrain_embed = torch.load(config.emb_file)
else:
    pretrain_embed = None

# model
print('building model...\n')
model = getattr(models, opt.model)(config, src_vocab.size(), tgt_vocab.size(), use_cuda,
                       pretrain=pretrain_embed, score_fn=opt.score) 

if opt.restore:
    model.load_state_dict(checkpoints['model'])
if use_cuda:
    model.cuda()
if len(opt.gpus) > 1:  
    model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)

# optimizer
if opt.restore:
    optim = checkpoints['optim']
else:
    optim = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                  lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
optim.set_parameters(model.parameters())
if config.schedule:
    scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)

# total number of parameters
param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]

if not os.path.exists(config.log):
    os.mkdir(config.log)
if opt.log == '':
    log_path = config.log + utils.format_time(time.localtime()) + '/'
else:
    log_path = config.log + opt.log + '/'
if not os.path.exists(log_path):
    os.mkdir(log_path)
logging = utils.logging(log_path+'log.txt') 
logging_csv = utils.logging_csv(log_path+'record.csv') 
for k, v in config.items():
    logging("%s:\t%s\n" % (str(k), str(v)))
logging("\n")
logging(repr(model)+"\n\n")  

logging('total number of parameters: %d\n\n' % param_count)
logging('score function is %s\n\n' % opt.score)

if opt.restore:
    updates = checkpoints['updates']
else:
    updates = 0

total_loss, start_time = 0, time.time()
report_total, report_correct = 0, 0
report_vocab, report_tot_vocab = 0, 0
scores = [[] for metric in config.metric]
scores = collections.OrderedDict(zip(config.metric, scores))

with open(opt.label_dict_file, 'r') as f:
    label_dict = json.load(f)

# train
def train(epoch):
    e = epoch
    model.train()

    if config.schedule:
        scheduler.step()
        print("Decaying learning rate to %g" % scheduler.get_lr()[0])

    if opt.model == 'gated': 
        model.current_epoch = epoch

    global e, updates, total_loss, start_time, report_total

    for raw_src, src, src_len, raw_tgt, tgt, tgt_len in trainloader:

        src = Variable(src)
        tgt = Variable(tgt)
        src_len = Variable(src_len).unsqueeze(0)
        tgt_len = Variable(tgt_len).unsqueeze(0)
        if use_cuda:
            src = src.cuda()
            tgt = tgt.cuda()
            src_len = src_len.cuda()
            tgt_len = tgt_len.cuda()

        model.zero_grad()
        outputs, targets = model(src, src_len, tgt, tgt_len)
        loss, num_total, _, _, _ = model.compute_loss(outputs, targets, opt.memory)

        total_loss += loss
        report_total += num_total
        optim.step()
        updates += 1  

        if updates % config.eval_interval == 0:
            logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.3f\n"
                    % (time.time()-start_time, epoch, updates, total_loss / report_total))
            print('evaluating after %d updates...\r' % updates)
            score = eval(epoch)
            for metric in config.metric:
                scores[metric].append(score[metric])
                if metric == 'micro_f1' and score[metric] >= max(scores[metric]):  
                    save_model(log_path+'best_'+metric+'_checkpoint.pt')
                if metric == 'hamming_loss' and score[metric] <= min(scores[metric]):
                    save_model(log_path+'best_'+metric+'_checkpoint.pt')

            model.train()
            total_loss = 0
            start_time = 0
            report_total = 0

        if updates % config.save_interval == 0:  
            save_model(log_path+'checkpoint.pt')


def eval(epoch):
    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    for raw_src, src, src_len, raw_tgt, tgt, tgt_len in validloader:
        if len(opt.gpus) > 1:
            samples, alignment = model.module.sample(src, src_len)
        else:
            samples, alignment = model.beam_sample(src, src_len, beam_size=config.beam_size)

        candidate += [tgt_vocab.convertToLabels(s, dict.EOS) for s in samples]
        source += raw_src
        reference += raw_tgt
        alignments += [align for align in alignment]

    if opt.unk:
        cands = []
        for s, c, align in zip(source, candidate, alignments):
            cand = []
            for word, idx in zip(c, align):
                if word == dict.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)
            cands.append(cand)
        candidate = cands

    score = {}
    result = utils.eval_metrics(reference, candidate, label_dict, log_path)
    logging_csv([e, updates, result['hamming_loss'], \
                result['micro_f1'], result['micro_precision'], result['micro_recall']])
    print('hamming_loss: %.8f | micro_f1: %.4f'
          % (result['hamming_loss'], result['micro_f1']))
    score['hamming_loss'] = result['hamming_loss']
    score['micro_f1'] = result['micro_f1']
    return score


def save_model(path):
    global updates
    model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates}
    torch.save(checkpoints, path)


def main():
    for i in range(1, config.epoch+1):
        if not opt.notrain:
            train(i)
        else:
            eval(i)
    for metric in config.metric:
        logging("Best %s score: %.2f\n" % (metric, max(scores[metric])))


if __name__ == '__main__':
    main()
