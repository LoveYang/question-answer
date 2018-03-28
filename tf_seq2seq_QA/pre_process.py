# -*- coding: utf-8 -*-
import os
from gensim.models import KeyedVectors
import numpy as np

dictionarypath = os.path.join(os.path.dirname(os.getcwd()), 'corpus', 'dictionary')
dictNameList = ['glove.42B.300d.txt', 'glove.6B.100d.txt']

dvcPath_F = lambda path: os.path.join(dictionarypath, path)
NewdvcPath_F = lambda path: os.path.join(dictionarypath, '_'.join(path.split('.')[:-1] + ['gensim']) + '.txt')

DEFAULT_DICT = 'glove_100D.txt'


def generatefile2gensimVW(path, dim):
    newdictName = '_'.join(path.split('.')[:-1] + ['gensim']) + '.txt'
    dvcPath = dvcPath_F(path)
    with open(dvcPath, 'rb') as rf:
        for linenum, line in enumerate(rf):
            pass
    linenum += 1
    with open(dvcPath, 'rb') as rf:
        with open(os.path.join(dictionarypath, newdictName), 'wb') as wf:
            wf.write("{0:d} {1:d}\n".format(linenum, dim).encode('utf8'))
            for lines in rf:
                wf.write(lines)


def generate_embeding_dictionary(keyVector, dictName="dictionary.txt"):
    if not isinstance(keyVector, KeyedVectors):
        raise TypeError("keyVector must be gensim.models.KeyedVectors")
    embedPath = os.path.join(os.getcwd(), 'embedding')
    if not os.path.exists(embedPath):
        os.mkdir(embedPath)

    embeddingName = '_'.join(dictName.split('.')[:-1] + ["embedding"])
    embedding = keyVector.syn0
    dictionary = [[indexobj.index, word] for word, indexobj in keyVector.vocab.items()]
    dictionary.sort(key=lambda x: x[0])
    np.save(os.path.join(embedPath, embeddingName), embedding)
    with open(os.path.join(embedPath, dictName), 'wb') as wf:
        for word in dictionary:
            wf.write(str(word[1] + '\n').encode("utf8"))


def getIND2word_WORD2ind_EMBEDDING(path='glove_100D.txt', decode='utf8', dictLoad=True, embeddingLoad=True):
    embedPath = os.path.join(os.getcwd(), 'embedding')
    if not os.path.exists(embedPath):
        raise NameError("no path {0:s) exist!}".format(embedPath))
    embeddingName = '_'.join(path.split('.')[:-1] + ["embedding"])
    ind2word = {}
    word2ind = {}
    embedding = None
    if embeddingLoad:
        embedding = np.load(os.path.join(embedPath, embeddingName + '.npy'))
    if dictLoad:
        with open(os.path.join(embedPath, path), 'rb') as rf:
            for index, line in enumerate(rf):
                word = line.decode(decode).strip()
                ind2word[index] = word
                word2ind[word] = index
    return ind2word, word2ind, embedding


class SimpleDict:
    def __init__(self, path=DEFAULT_DICT, start='<s>', end='</s>', unk='<unk>', pad='<pad>', dictLoad=True,
                 embeddingLoad=False):
        self.path = path
        ind2word, word2ind, _ = getIND2word_WORD2ind_EMBEDDING(path, dictLoad=dictLoad, embeddingLoad=False)
        vocab_size = len(ind2word)
        createNew = 0
        if word2ind.get(start) is None:
            word2ind[start] = vocab_size
            ind2word[vocab_size] = start
            vocab_size += 1
            createNew += 1
        elif vocab_size - word2ind.get(start) > 4:
            raise ValueError("{0:s} must be last-3 in sequence".format(start))

        if word2ind.get(end) is None:
            word2ind[end] = vocab_size
            ind2word[vocab_size] = end
            vocab_size += 1
            createNew += 1
        elif vocab_size - word2ind.get(end) > 4:
            raise ValueError("{0:s} must be last-3 in sequence".format(end))
        if word2ind.get(pad) is None:
            word2ind[pad] = vocab_size
            ind2word[vocab_size] = pad
            vocab_size += 1
            createNew += 1
        elif vocab_size - word2ind.get(pad) > 4:
            raise ValueError("{0:s} must be last-3 in sequence".format(pad))

        if word2ind.get(unk) is None:
            word2ind[unk] = vocab_size
            ind2word[vocab_size] = unk
            vocab_size += 1
            createNew += 1
        elif vocab_size - word2ind.get(unk) > 4:
            raise ValueError("{0:s} must be last-3 in sequence".format(unk))
        self.embedding_shape = None
        self.embedding = None

        self.createNew = createNew
        self.ind2word, self.word2ind = ind2word, word2ind
        self.vocab_size = vocab_size
        self.start = start
        self.end = end
        self.unk = unk
        self.pad = pad

        if embeddingLoad:
            self._loadsEmbed()

    def _loadsEmbed(self):
        _, _, embedding = getIND2word_WORD2ind_EMBEDDING(self.path, dictLoad=False, embeddingLoad=True)
        embedding_shape = embedding.shape
        if self.createNew > 0:
            self.embedding = np.concatenate([embedding, np.random.normal(size=[self.createNew, embedding_shape[1]])],
                                            axis=0)
        else:
            self.embedding = embedding
        self.embedding_shape = embedding.shape

    def Transform_word2ind(self, seq):
        return list(map(lambda x: self.word2ind.get(x.lower()), seq))

    def Transform_ind2word(self, seq):
        return list(map(lambda x: self.ind2word.get(x), seq))

    def _Transform_word2indReplaceNone2UNK(self, seq):
        return list(map(lambda x: self.word2ind.get(x.lower()) or self.word2ind.get(self.unk), seq))

    def BatchSizeAddStartSign(self, arr):
        return np.column_stack((np.ones([arr.shape[0], 1], dtype=np.int32) * self.word2ind[self.start], arr))

    def BatchSizeMaxLen2END(self, arr):
        return np.argwhere(arr == self.word2ind[self.end])[:, 1] + 1

    def Transform_sentence2ind(self, corpus, padLength=20, addStart=False, addEnd=True):
        assert (all(map(lambda x: isinstance(x, list), corpus)))
        corpusInd = []
        for sentence in corpus:
            temp = self._Transform_word2indReplaceNone2UNK(sentence)
            if addStart:
                temp = [self.word2ind[self.start]] + temp
            if addEnd:
                temp.append(self.word2ind[self.end])
            ls = len(temp)
            if ls > padLength:
                raise ValueError("length of sentence: {0:s} greater than padLength{1:d}".format(sentence, padLength))
            temp = np.pad(np.array(temp, dtype=np.int32), (0, padLength - ls), mode="constant",
                          constant_values=(self.word2ind[self.start], self.word2ind[self.pad]))
            corpusInd.append(temp)
        return np.row_stack(corpusInd)


DataPATH = os.path.join(os.getcwd(), 'data')
if not os.path.exists(DataPATH):
    os.mkdir(DataPATH)


def indexDataSave(arr, path):
    np.save(os.path.join(DataPATH, path), arr)


def indexDataLoad(path):
    return np.load(os.path.join(DataPATH, path) + '.npy')


if __name__ == "__main__":
    # generatefile2gensimVW('glove.6B.100d.txt', 100)
    # dvcPath = NewdvcPath_F('glove.6B.100d.txt')
    # ans = KeyedVectors.load_word2vec_format(dvcPath)
    # generate_embeding_dictionary(ans, dictName='glove_100D.txt')
    # ind2word, word2ind, embedding = getIND2word_WORD2ind_EMBEDDING('glove_100D.txt')
    from tf_seq2seq_QA.corpus_parse import run_test

    qs = run_test()
    dt = SimpleDict('glove_100D.txt')
    ParaMaxLength = max(list(map(lambda x: len(x[1]), qs)))
    QuesMaxLength = max(list(map(lambda x: len(x[0]), qs)))
    questionind = dt.Transform_sentence2ind([x[0] for x in qs], QuesMaxLength + 1, addStart=False)
    paragraphind = dt.Transform_sentence2ind([x[1] for x in qs], ParaMaxLength + 1, addStart=False)
    indexDataSave(questionind, 'SQuAD_Question_index_glove_100D')
    indexDataSave(paragraphind, 'SQuAD_Paragraphy_index_glove_100D')
