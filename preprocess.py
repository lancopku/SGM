import os
import argparse
import codecs
import utils
import pickle


parser = argparse.ArgumentParser(description='preprocess.py')

parser.add_argument('-load_data', type=str, required=True,
                    help="input file dir for the data")
parser.add_argument('-save_data', type=str, required=True,
                    help="output file dir for the processed data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    help="size of the target vocabulary")
parser.add_argument('-src_filter', type=int, default=0,
                    help="maximum source sequence length")
parser.add_argument('-tgt_filter', type=int, default=0,
                    help="maximum target sequence length")
parser.add_argument('-src_trun', type=int, default=0,
                    help="truncate source sequence length")
parser.add_argument('-tgt_trun', type=int, default=0,
                    help="truncate target sequence length")
parser.add_argument('-src_char', action='store_true', 
                    help='character based encoding')
parser.add_argument('-tgt_char', action='store_true', 
                    help='character based decoding')
parser.add_argument('-src_suf', default='src',
                    help="the suffix of the source filename")
parser.add_argument('-tgt_suf', default='tgt',
                    help="the suffix of the target filename")

parser.add_argument('-share', action='store_true', 
                    help='share the vocabulary between source and target')

parser.add_argument('-report_every', type=int, default=100000,
                    help="report status every this many sentences")

opt = parser.parse_args()


def makeVocabulary(filename, trun_length, filter_length, char, vocab, size):

    print("%s: length limit = %d, truncate length = %d" % (filename, filter_length, trun_length))
    max_length = 0
    with codecs.open(filename, 'r', 'utf-8') as f:
        for sent in f.readlines():
            if char:
                tokens = list(sent.strip())
            else:
                tokens = sent.strip().split()
            if 0 < filter_length < len(sent.strip().split()):
                continue
            max_length = max(max_length, len(tokens))
            if trun_length > 0:
                tokens = tokens[:trun_length]
            for word in tokens:
                vocab.add(word)

    print('Max length of %s = %d' % (filename, max_length))

    if size > 0:
        originalSize = vocab.size()
        vocab = vocab.prune(size)
        print('Created dictionary of size %d (pruned from %d)' %
              (vocab.size(), originalSize))

    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, tgtFile, srcDicts, tgtDicts, save_srcFile, save_tgtFile, lim=0):
    sizes = 0
    count, empty_ignored, limit_ignored = 0, 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile, encoding='utf8')
    tgtF = open(tgtFile, encoding='utf8')

    srcIdF = open(save_srcFile + '.id', 'w')
    tgtIdF = open(save_tgtFile + '.id', 'w')
    srcStrF = open(save_srcFile + '.str', 'w', encoding='utf8')
    tgtStrF = open(save_tgtFile + '.str', 'w', encoding='utf8')

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
            empty_ignored += 1
            continue

        sline = sline.lower()
        tline = tline.lower()

        srcWords = sline.split() if not opt.src_char else list(sline)
        tgtWords = tline.split() if not opt.tgt_char else list(tline)


        if (opt.src_filter == 0 or len(sline.split()) <= opt.src_filter) and \
           (opt.tgt_filter == 0 or len(tline.split()) <= opt.tgt_filter):

            if opt.src_trun > 0:
                srcWords = srcWords[:opt.src_trun]
            if opt.tgt_trun > 0:
                tgtWords = tgtWords[:opt.tgt_trun]

            srcIds = srcDicts.convertToIdx(srcWords, utils.UNK_WORD)
            tgtIds = tgtDicts.convertToIdx(tgtWords, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD)

            srcIdF.write(" ".join(list(map(str, srcIds)))+'\n')
            tgtIdF.write(" ".join(list(map(str, tgtIds)))+'\n')
            if not opt.src_char:
                srcStrF.write(" ".join(srcWords)+'\n')
            else:
                srcStrF.write("".join(srcWords) + '\n')
            if not opt.tgt_char:
                tgtStrF.write(" ".join(tgtWords)+'\n')
            else:
                tgtStrF.write("".join(tgtWords) + '\n')

            sizes += 1
        else:
            limit_ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()
    srcStrF.close()
    tgtStrF.close()
    srcIdF.close()
    tgtIdF.close()

    print('Prepared %d sentences (%d and %d ignored due to length == 0 or > )' %
          (sizes, empty_ignored, limit_ignored))

    return {'srcF': save_srcFile + '.id', 'tgtF': save_tgtFile + '.id',
            'original_srcF': save_srcFile + '.str', 'original_tgtF': save_tgtFile + '.str',
            'length': sizes}


def main():

    if not os.path.exists(opt.save_data):
        os.makedirs(opt.save_data)

    dicts = {}

    train_src, train_tgt = opt.load_data + 'train.' + opt.src_suf, opt.load_data + 'train.' + opt.tgt_suf
    valid_src, valid_tgt = opt.load_data + 'valid.' + opt.src_suf, opt.load_data + 'valid.' + opt.tgt_suf
    test_src, test_tgt = opt.load_data + 'test.' + opt.src_suf, opt.load_data + 'test.' + opt.tgt_suf

    save_train_src, save_train_tgt = opt.save_data + 'train.' + opt.src_suf, opt.save_data + 'train.' + opt.tgt_suf
    save_valid_src, save_valid_tgt = opt.save_data + 'valid.' + opt.src_suf, opt.save_data + 'valid.' + opt.tgt_suf
    save_test_src, save_test_tgt = opt.save_data + 'test.' + opt.src_suf, opt.save_data + 'test.' + opt.tgt_suf

    src_dict, tgt_dict = opt.save_data + 'src.dict', opt.save_data + 'tgt.dict'

    if opt.share:
        assert opt.src_vocab_size == opt.tgt_vocab_size
        print('Building source and target vocabulary...')
        dicts['src'] = dicts['tgt'] = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD])
        dicts['src'] = makeVocabulary(train_src, opt.src_trun, opt.src_filter, opt.src_char, dicts['src'], opt.src_vocab_size)
        dicts['src'] = dicts['tgt'] = makeVocabulary(train_tgt, opt.tgt_trun, opt.tgt_filter, opt.tgt_char, dicts['tgt'], opt.tgt_vocab_size)
    else:
        print('Building source vocabulary...')
        dicts['src'] = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD])
        dicts['src'] = makeVocabulary(train_src, opt.src_trun, opt.src_filter, opt.src_char, dicts['src'], opt.src_vocab_size)
        print('Building target vocabulary...')
        dicts['tgt'] = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD])
        dicts['tgt'] = makeVocabulary(train_tgt, opt.tgt_trun, opt.tgt_filter, opt.tgt_char, dicts['tgt'], opt.tgt_vocab_size)

    print('Preparing training ...')
    train = makeData(train_src, train_tgt, dicts['src'], dicts['tgt'], save_train_src, save_train_tgt)

    print('Preparing validation ...')
    valid = makeData(valid_src, valid_tgt, dicts['src'], dicts['tgt'], save_valid_src, save_valid_tgt)

    print('Preparing test ...')
    test = makeData(test_src, test_tgt, dicts['src'], dicts['tgt'], save_test_src, save_test_tgt)

    print('Saving source vocabulary to \'' + src_dict + '\'...')
    dicts['src'].writeFile(src_dict)

    print('Saving source vocabulary to \'' + tgt_dict + '\'...')
    dicts['tgt'].writeFile(tgt_dict)

    data = {'train': train, 'valid': valid,
             'test': test, 'dict': dicts}
    pickle.dump(data, open(opt.save_data+'data.pkl', 'wb'))


if __name__ == "__main__":
    main()