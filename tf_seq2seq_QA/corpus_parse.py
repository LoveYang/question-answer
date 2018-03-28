import os
import json
from nltk.tokenize import word_tokenize
import nltk

try:
    word_tokenize(' ')
except LookupError:
    nltk.download('punkt')


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


class Question2ParaAndDoc:
    def __init__(self, dict_DocSQuAD):
        if not isinstance(dict_DocSQuAD, DocSQuAD):
            raise TypeError("param of Question2ParaAndDoc must be DocSQuAD")
        content = dict_DocSQuAD.content
        self.titles = dict([(ind, context) for ind, context in content.items() if len(ind.split(',')) == 1])
        self.paragraphs = dict([(ind, context) for ind, context in content.items() if len(ind.split(',')) == 2])
        self.questions = dict([(ind, context) for ind, context in content.items() if len(ind.split(',')) == 3])
        self.answers = dict([(ind, context) for ind, context in content.items() if len(ind.split(',')) == 4])

    @property
    def _outQ2P(self):
        x2y = []
        for qs in self.questions.items():
            x2y.append([qs[1], self.paragraphs[','.join(qs[0].split(',')[:2])]])
        return x2y

    @property
    def Q2PToken(self):
        Q2PToken = []
        for q, p in self._outQ2P:
            Q2PToken.append((word_tokenize(q, language='english'), word_tokenize(p, language='english')))
        return Q2PToken


def run_test():
    xtrainPath = 'train-v1.1.json'
    workpath = os.path.join(os.path.dirname(os.getcwd()), 'corpus', 'SQuAD')
    with open(os.path.join(workpath, xtrainPath), 'r', encoding='utf8') as rf:
        rcontent = json.loads(rf.read())
    # load file
    corpus = DocSQuAD(rcontent)
    qs = Question2ParaAndDoc(corpus)
    xy_qs = qs.Q2PToken
    return xy_qs


if __name__ == "__main__":
    xy_qs = run_test()
