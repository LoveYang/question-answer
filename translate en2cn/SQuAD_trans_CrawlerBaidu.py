from trans_en2cn import translateEN2ZH, logger
import os
import json


class DocSQuAD:
    '''
    -file.json
        -data
            -[i]
                -paragraphs
                    -[i]
                        -context [str]
                        -qas
                            -[i]
                                -answers
                                    -[i]
                                        -answer_start [int]
                                        -text [str]
                                -id [str]
                                -question [str]

                -title
        -version
    '''

    def __init__(self, DOCcontent):
        self.content = {}
        self.ToSource = DOCcontent
        self._analysis_Content(DOCcontent)

    def _analysis_Content(self, DOCcontent):
        for docIndex, doc in enumerate(self._isSQuADText(DOCcontent, 'data')):
            for paraIndex, para in enumerate(self._isSQuADText(doc, 'paragraphs')):
                self.content[','.join([str(docIndex), str(paraIndex)])] = self._isSQuADText(para, 'context')
                for qasIndex, qas in enumerate(self._isSQuADText(para, 'qas')):
                    self.content[','.join([str(docIndex), str(paraIndex), str(qasIndex)])] = self._isSQuADText(qas,
                                                                                                               'question')
                    for ansIndex, ans in enumerate(self._isSQuADText(qas, 'answers')):
                        self.content[','.join(
                            [str(docIndex), str(paraIndex), str(qasIndex), str(ansIndex)])] = self._isSQuADText(ans,
                                                                                                                'text')

    def getNewDoc(self, content=None):
        if content is None:
            content = self.content
        newDoc = self.ToSource.copy()
        for docIndex, doc in enumerate(self._isSQuADText(newDoc, 'data')):
            for paraIndex, para in enumerate(self._isSQuADText(doc, 'paragraphs')):
                para['context'] = content[','.join([str(docIndex), str(paraIndex)])]
                for qasIndex, qas in enumerate(self._isSQuADText(para, 'qas')):
                    qas['question'] = content[','.join([str(docIndex), str(paraIndex), str(qasIndex)])]
                    for ansIndex, ans in enumerate(self._isSQuADText(qas, 'answers')):
                        ans['text'] = content[','.join([str(docIndex), str(paraIndex), str(qasIndex), str(ansIndex)])]
        return newDoc

    def _isSQuADText(self, CT, pathStr, limitPAth=None):
        '''
        :param CT: Doc
        :param pathStr: version,data, title,paragraphs, context,qas,answers,id,question ,answer_start,text
        :return: item
        '''
        if limitPAth is None:
            limitPAth = ['version', 'data', 'title', 'paragraphs', 'context', 'qas'
                , 'answers', 'id', 'question', 'answer_start', 'text']
        if pathStr in limitPAth:
            return CT.get(pathStr)
        else:
            raise KeyError("%s is not in limitPAth:%s" % (pathStr, limitPAth))


import time


def parse_dict(x):
    curTime = time.time()
    ans = (x[0], translateEN2ZH(x[1]))
    print('....: ', x[0], ' __custom time :', time.time() - curTime)
    return ans


if __name__ == "__main__":
    from multiprocessing import Pool

    pool = Pool(os.cpu_count())
    workpath = os.path.join(os.path.dirname(os.getcwd()), 'corpus', 'SQuAD')
    sourcelist = ['dev-v1.1.json', 'train-v1.1.json']
    sourcelist = [sourcelist[0]]
    for singleSource in sourcelist:
        with open(os.path.join(workpath, singleSource), 'r', encoding='utf8') as rf:
            rcontent = json.loads(rf.read())
            doc = DocSQuAD(rcontent)
            en2zhCache = dict(pool.map(parse_dict, doc.content.items()))
            with open(os.path.join(workpath, "EN2ZH_CACHE_" + singleSource), 'w', encoding='utf8') as wf:
                wf.write(json.dumps(en2zhCache, ensure_ascii=False))
            en2zhDoc = doc.getNewDoc(en2zhCache)

        with open(os.path.join(workpath, "EN2ZH_" + singleSource), 'w', encoding='utf8') as wf:
            wf.write(json.dumps(en2zhDoc, ensure_ascii=False))
