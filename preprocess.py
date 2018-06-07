import argparse
import torch
import data.dict as dict
from data.dataloader import dataset

parser = argparse.ArgumentParser(description='preprocess.py')

##
## **Preprocess Options**
##

parser.add_argument('-config', help="Read options from this file")

parser.add_argument('-train_src', 
                    default='./data/data/text_train',
                    help="Path to the training source data")
parser.add_argument('-train_tgt', 
                    default='./data/data/label_train',
                    help="Path to the training target data")
parser.add_argument('-valid_src',
                    default='./data/data/text_val',
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', 
                    default='./data/data/label_val',
                    help="Path to the validation target data")
parser.add_argument('-test_src', 
                    default='./data/data/text_test',
                    help="Path to the validation source data")
parser.add_argument('-test_tgt', 
                    default='./data/data/label_test',
                    help="Path to the validation target data")

parser.add_argument('-save_data', 
                    default='./data/data/save_data',
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=150,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab', default=None,
                    help="Path to an existing source vocabulary") 
parser.add_argument('-tgt_vocab', default=None,
                    help="Path to an existing target vocabulary")


parser.add_argument('-src_length', type=int, default=500,
                    help="Maximum source sequence length")
parser.add_argument('-tgt_length', type=int, default=25,
                    help="Maximum target sequence length")
parser.add_argument('-shuffle',    type=int, default=0,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', default = True, action='store_true', help='lowercase data')
parser.add_argument('-char', default = False, action='store_true', help='replace unk with char')
parser.add_argument('-share', default = False, action='store_true', help='share the vocabulary between source and target')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()
torch.manual_seed(opt.seed)

def makeVocabulary(filename, size, char=False): 
    vocab = dict.Dict([dict.PAD_WORD, dict.UNK_WORD,
                       dict.BOS_WORD, dict.EOS_WORD], lower=opt.lower)
    if char:
        vocab.addSpecial(dict.SPA_WORD)

    lengths = []

    if type(filename) == list:
        for _filename in filename:
            with open(_filename) as f:
                for sent in f.readlines():
                    for word in sent.strip().split():
                        lengths.append(len(word))
                        if char:
                            for ch in word:
                                vocab.add(ch)
                        else:
                            vocab.add(word + " ") 
    else:
        with open(filename) as f:
            for sent in f.readlines():
                for word in sent.strip().split():
                    lengths.append(len(word))
                    if char:
                        for ch in word:
                            vocab.add(ch)
                    else:
                        vocab.add(word+" ")

    print('max: %d, min: %d, avg: %.2f' % (max(lengths), min(lengths), sum(lengths)/len(lengths)))

    originalSize = vocab.size()
    vocab = vocab.prune(size) 
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize, char=False):
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = dict.Dict()
        vocab.loadFile(vocabFile) 
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFile, vocabSize, char=char) 
        vocab = genWordVocab

    print()
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, tgtFile, srcDicts, tgtDicts, sort=False, char=False):
    src, tgt = [], [] 
    raw_src, raw_tgt = [], [] 
    sizes = [] 
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile)
    tgtF = open(tgtFile)

    while True: 
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            ignored += 1
            continue

        if opt.lower:
            sline = sline.lower()
            tline = tline.lower()

        srcWords = sline.split()
        tgtWords = tline.split()

        # 
        if opt.src_length == 0 or (len(srcWords) <= opt.src_length and len(tgtWords) <= opt.tgt_length):

            if char:
                srcWords = [word + " " for word in srcWords]
                tgtWords = list(" ".join(tgtWords))
            else:
                srcWords = [word+" " for word in srcWords]
                tgtWords = [word+" " for word in tgtWords]

            src += [srcDicts.convertToIdx(srcWords,
                                          dict.UNK_WORD)] 
            tgt += [tgtDicts.convertToIdx(tgtWords,
                                          dict.UNK_WORD,
                                          dict.BOS_WORD,
                                          dict.EOS_WORD)]
            raw_src += [srcWords]
            raw_tgt += [tgtWords]
            sizes += [len(srcWords)] 
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]
        raw_src = [raw_src[idx] for idx in perm]
        raw_tgt = [raw_tgt[idx] for idx in perm]

    if sort:
        print('... sorting sentences by size')
        _, perm = torch.sort(torch.Tensor(sizes))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        raw_src = [raw_src[idx] for idx in perm]
        raw_tgt = [raw_tgt[idx] for idx in perm]

    print('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
          (len(src), ignored, opt.src_length))

    return dataset(src, tgt, raw_src, raw_tgt) 


def main():
    dicts = {}
    if opt.share:
        assert opt.src_vocab_size == opt.tgt_vocab_size
        print('share the vocabulary between source and target')
        dicts['src'] = initVocabulary('source and target',
                                      [opt.train_src, opt.train_tgt],
                                      opt.src_vocab,
                                      opt.src_vocab_size)
        dicts['tgt'] = dicts['src']
    else:
        dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,
                                      opt.src_vocab_size)
        dicts['tgt'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab,
                                      opt.tgt_vocab_size, char=opt.char)

    print('Preparing training ...')
    train = makeData(opt.train_src, opt.train_tgt, dicts['src'], dicts['tgt'], char=opt.char)

    print('Preparing validation ...')
    valid = makeData(opt.valid_src, opt.valid_tgt, dicts['src'], dicts['tgt'], char=opt.char)

    print('Preparing test ...')
    test = makeData(opt.test_src, opt.test_tgt, dicts['src'], dicts['tgt'], char=opt.char)

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')

    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': train,
                 'valid': valid,
                 'test': test}
    torch.save(save_data, opt.save_data) 


if __name__ == "__main__":
    main()