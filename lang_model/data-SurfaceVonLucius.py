import torch
import urllib.request


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize('https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt')
        self.valid = self.tokenize('https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt')
        self.test = self.tokenize('https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt')

    def tokenize(self, path):
        """Tokenizes a text file."""

        # Add words to the dictionary
        with urllib.request.urlopen(path) as f:
            for lineraw in f:
                line = lineraw.decode('utf-8')
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with urllib.request.urlopen(path) as f:
            idss = []
            for lineraw in f:
                line = lineraw.decode('utf-8')
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

