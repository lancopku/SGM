import torch
from collections import OrderedDict

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk> '
BOS_WORD = '<s>'
EOS_WORD = '</s>'


class Dict(object):
    def __init__(self, data=None, lower=True):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = {}
        self.lower = lower
        # Special entries will not be pruned.
        self.special = []

        if data is not None:
            if type(data) == str:
                self.loadFile(data)
            else:
                self.addSpecials(data)

    def size(self):
        return len(self.idxToLabel)

    # Load entries from a file.
    def loadFile(self, filename):
        for line in open(filename):
            fields = line.split()
            label = fields[0]
            idx = int(fields[1])
            self.add(label, idx)

    # Write entries to a file.
    def writeFile(self, filename):
        with open(filename, 'w') as file:
            for i in range(self.size()):
                label = self.idxToLabel[i]
                file.write('%s %d\n' % (label, i))

        file.close()

    def loadDict(self, idxToLabel):
        for i in range(len(idxToLabel)):
            label = idxToLabel[i]
            self.add(label, i)

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default

    def getLabel(self, idx, default=None):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default

    # Mark this `label` and `idx` as special (i.e. will not be pruned).
    def addSpecial(self, label, idx=None):
        idx = self.add(label, idx)
        self.special += [idx]

    # Mark all labels in `labels` as specials (i.e. will not be pruned).
    def addSpecials(self, labels):
        for label in labels:
            self.addSpecial(label)

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label, idx=None):
        label = label.lower() if self.lower else label
        if idx is not None:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        else:
            if label in self.labelToIdx:
                idx = self.labelToIdx[label]
            else:
                idx = len(self.idxToLabel)
                self.idxToLabel[idx] = label
                self.labelToIdx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    # Return a new dictionary with the `size` most frequent entries.
    def prune(self, size):
        if size > self.size():
            return self

        # Only keep the `size` most frequent entries.
        freq = torch.tensor(
                [self.frequencies[i] for i in range(len(self.frequencies))])
        _, idx = torch.sort(freq, 0, True)
        idx = idx.tolist()

        newDict = Dict()
        newDict.lower = self.lower

        # Add special entries in all cases.
        for i in self.special:
            newDict.addSpecial(self.idxToLabel[i])

        for i in idx[:size]:
            newDict.add(self.idxToLabel[i])

        return newDict

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertToIdx(self, labels, unkWord, bosWord=None, eosWord=None):
        vec = []

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)
        vec += [self.lookup(label, default=unk) for label in labels]

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        return vec

    def convertToIdxandOOVs(self, labels, unkWord, bosWord=None, eosWord=None):
        vec = []
        oovs = OrderedDict()

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)
        for label in labels:
            id = self.lookup(label, default=unk)
            if id != unk:
                vec += [id]
            else:
                if label not in oovs:
                    oovs[label] = len(oovs)+self.size()
                oov_num = oovs[label]
                vec += [oov_num]

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        return torch.LongTensor(vec), oovs

    def convertToIdxwithOOVs(self, labels, unkWord, bosWord=None, eosWord=None, oovs=None):
        vec = []

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)
        for label in labels:
            id = self.lookup(label, default=unk)
            if id == unk and label in oovs:
                vec += [oovs[label]]
            else:
                vec += [id]

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        return torch.LongTensor(vec)

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def convertToLabels(self, idx, stop, oovs=None):
        labels = []

        for i in idx:
            if i == stop:
                break
            if i < self.size():
                labels += [self.getLabel(i)]
            else:
                labels += [oovs[i-self.size()]]

        return labels
