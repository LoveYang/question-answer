# -*- coding: utf-8 -*-

import numpy as np
import time


class dataSet:
    def __init__(self, data, batch_size, train_rate_valid=0.8):
        self.batch_size = batch_size
        assert (train_rate_valid <= 1)
        if isinstance(data, tuple):
            [np.random.shuffle(x) for x in data]
            self.data = data
        else:
            raise TypeError("data must be tuple")
        datashape = [x.shape[0] for x in data]
        if False in [x == y for x in datashape for y in datashape]:
            raise ValueError("all data must be same shape", datashape)
        else:
            self.sizeAxiszero = datashape[0]
        self.next_batch_index = 0
        self.trainsize = int(np.round(self.sizeAxiszero * train_rate_valid))
        self.validsize = int(self.sizeAxiszero - self.trainsize)
        self.traindata = [x[:self.trainsize, :] for x in self.data]
        self.validdata = [x[self.trainsize:, :] for x in self.data]

    def next_batch_train(self):
        next_batch_ind = self.next_batch_index + self.batch_size
        curdata = [x[self.next_batch_index:next_batch_ind, :] if x.shape[0] > next_batch_ind else None for x in
                   self.traindata]
        self.next_batch_index = next_batch_ind
        return curdata

    def next_batch_resample_train(self):
        ind = np.random.choice(np.arange(self.trainsize), self.batch_size)
        return [x[ind, :] for x in self.traindata]

    def next_batch_resample_valid(self):
        ind = np.random.choice(np.arange(self.validsize), self.batch_size)
        return [x[ind, :] for x in self.validdata]


class TimeTick:
    def __init__(self):
        self._start()

    def _start(self):
        self.curtime = time.time()

    def _nexttime(self):
        newtime = time.time()
        self.timegap = newtime - self.curtime
        self.curtime = newtime

    @property
    def time(self):
        self._nexttime()
        print("run time is %s s" % (self.timegap))
        return self.timegap


if __name__ == "__main__":
    from tf_seq2seq_QA.pre_process import DataPATH, indexDataLoad, SimpleDict

    qsdata = indexDataLoad('SQuAD_Question_index_glove_100D')
    paradata = indexDataLoad("SQuAD_Paragraphy_index_glove_100D")
    dt = SimpleDict()
    decode_input = dt.BatchSizeAddStartSign(qsdata)
    print(decode_input[:10, :])
    data = dataSet((paradata, qsdata, decode_input), 32)

    print(data.next_batch_resample_train())
