import torch
import utils

class Beam(object):
    
    def __init__(self, size, n_best=1, cuda=True, length_norm=False, minimum_length=0):

        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(utils.EOS)]
        self.nextYs[0][0] = utils.BOS

        # Has EOS topped the beam yet.
        self._eos = utils.EOS
        self.eosTop = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        self.length_norm = length_norm
        self.minimum_length = minimum_length

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        return self.nextYs[-1]

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk, attnOut):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
            ngrams = []
            le = len(self.nextYs)
            for j in range(self.nextYs[-1].size(0)):
                hyp, _ = self.getHyp(le-1, j)
                ngrams = set()
                fail = False
                gram = []
                for i in range(le-1):
                    # last n tokens, n = block_ngram_repeat
                    gram = (gram + [hyp[i]])[-3:]
                    # skip the blocking if it is in the exclusion list
                    #if set(gram) & self.exclusion_tokens:
                        #continue
                    if tuple(gram) in ngrams:
                        fail = True
                    ngrams.add(tuple(gram))
                if fail:
                    beamLk[j] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.allScores.append(self.scores)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))
        self.attn.append(attnOut.index_select(0, prevK))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                if self.length_norm:
                    s /= len(self.nextYs)
                if len(self.nextYs) - 1 >= self.minimum_length:
                    self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == utils.EOS:
            self.allScores.append(self.scores)
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.n_best

    def beam_update(self, state, idx):
        positions = self.getCurrentOrigin()
        for e in state:
            a, br, d = e.size()
            e = e.view(a, self.size, br // self.size, d)
            sentStates = e[:, :, idx]
            sentStates.copy_(sentStates.index_select(1, positions))

    def beam_update_gru(self, state, idx):
        positions = self.getCurrentOrigin()
        for e in state:
            br, d = e.size()
            e = e.view(self.size, br // self.size, d)
            sentStates = e[:, idx]
            sentStates.copy_(sentStates.index_select(0, positions))

    def beam_update_memory(self, state, idx):
        positions = self.getCurrentOrigin()
        e = state
        br, d = e.size()
        e = e.view(self.size, br // self.size, d)
        sentStates = e[:, idx]
        sentStates.copy_(sentStates.index_select(0, positions))

    def sortFinished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                s = self.scores[i].item()
                self.finished.append((s, len(self.nextYs) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def getHyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
            hyp.append(self.nextYs[j+1][k].item())
            attn.append(self.attn[j][k])
            k = self.prevKs[j][k].item()
        return hyp[::-1], torch.stack(attn[::-1])
