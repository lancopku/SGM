import torch 
import torch.utils.data 

import os 
import argparse  
import pickle  
import codecs  
import json  
import random  
import numpy as np  

import opts  
import models  
import utils  


parser = argparse.ArgumentParser(description='predict.py')

opts.model_opts(parser)
parser.add_argument('-data', type=str, default='./data/save_data/', 
                    help="the processed data dir")
parser.add_argument('-batch_size', type=int, default=64,
                    help="the batch size for testing")

opt = parser.parse_args()


if not os.path.exists(opt.log):
    os.makedirs(opt.log)

# load checkpoint
assert opt.restore
print('loading checkpoint...\n')
checkpoints = torch.load(opt.restore)
config = checkpoints['config']

# set seed
torch.manual_seed(opt.seed)
random.seed(opt.seed)
np.random.seed(opt.seed)

# set cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
config.use_cuda = use_cuda
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True

# load label_dict
with codecs.open(opt.label_dict_file, 'r', 'utf-8') as f:
    label_dict = json.load(f)


def load_data():
    print('loading data...\n')
    data = pickle.load(open(opt.data+'data.pkl', 'rb'))
    src_vocab = data['dict']['src']
    tgt_vocab = data['dict']['tgt']
    config.src_vocab_size = src_vocab.size()
    config.tgt_vocab_size = tgt_vocab.size()
    testset = utils.BiDataset(data['test'], char=config.char) 
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             collate_fn=utils.padding) 
    return {'testset':testset, 'testloader': testloader,
            'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}  


# load data
data = load_data()

# build model
print('building model...\n')
model = getattr(models, opt.model)(config)
model.load_state_dict(checkpoints['model'])
if use_cuda:
    model.cuda()


def eval_model(model, data):

    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    tgt_vocab = data['tgt_vocab']
    count, total_count = 0, len(data['testset'])
    dataloader = data['testloader']

    for src, tgt, src_len, tgt_len, original_src, original_tgt in dataloader:

        if config.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()

        with torch.no_grad():
            if config.beam_size > 1 and (not config.global_emb):
                samples, alignment, _ = model.beam_sample(src, src_len, beam_size=config.beam_size, eval_=True)
            else:
                samples, alignment = model.sample(src, src_len)

        candidate += [tgt_vocab.convertToLabels(s.tolist(), utils.EOS) for s in samples]
        source += original_src
        reference += original_tgt
        if alignment is not None:
            alignments += [align for align in alignment]

        count += len(original_src)
        utils.progress_bar(count, total_count)

    if config.unk and config.attention != 'None':
        cands = []
        for s, c, align in zip(source, candidate, alignments):
            cand = []
            for word, idx in zip(c, align):
                if word == utils.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)
            cands.append(cand)
            if len(cand) == 0:
                print('Error!')
        candidate = cands

    results = utils.eval_metrics(reference, candidate, label_dict, opt.log)
    results = [('%s: %.5f' % item + '\n') for item in results.items()]
    with codecs.open(opt.log+'results.txt', 'w', 'utf-8') as f:
        f.writelines(results)


if __name__ == '__main__':
    eval_model(model, data)