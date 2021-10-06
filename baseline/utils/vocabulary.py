import spacy


class Vocabulary:
    def __init__(self, voc_file):
        self.words = [x.replace("\n", "") for x in open(voc_file, "r").readlines()]
        self.max_length = int(self.words[0])
        self.words = self.words[1:]
        self.w2ix = {}
        self.ix2w = {}
        self.nlp = spacy.load("en_core_web_sm")

        for i, w in enumerate(self.words):
            self.w2ix[w] = i
            self.ix2w[i] = w

    @property
    def pad_token_ix(self):
        return self.w2ix["<pad>"]

    @property
    def start_token_ix(self):
        return self.w2ix["<sos>"]

    @property
    def end_token_ix(self):
        return self.w2ix["<eos>"]

    @property
    def number_of_words(self):
        return len(self.words)

    def tokenize(self, sentence):
        return [x.text for x in self.nlp(sentence)]

    def sent2ix(self, sentence):
        return [self.w2ix[x.text] for x in self.nlp(sentence)]

    def sent2ix_andpad(self, sentence, add_eos_token=False):
        new = [self.w2ix[x.text] for x in self.nlp(sentence)]
        max_length = self.max_length
        if add_eos_token:
            max_length += 1
            new.append(self.end_token_ix)

        while len(new) < max_length:
            new.append(self.pad_token_ix)
        return new

    def ix2sent(self, indices):
        return [self.ix2w[ix] for ix in indices]

    def ix2sent_drop_pad(self, indices):
        sent = []
        for i in indices:
            w = self.ix2w[i]
            if w == "<pad>":
                return sent
            sent.append(w)
        return sent

    def compare_gt_with_pred(self, gt, predicted):
        for (g, p) in zip(gt, predicted):
            print("------")
            # print(g)
            print("gt:" + " ".join(self.ix2sent_drop_pad(g)))
            print("pr:" + " ".join(self.ix2sent_drop_pad(p)))
