import requests
import random
from hashlib import md5
import logging
import os

"""
log
"""
logpath = os.path.join(os.getcwd(), 'log')
if not os.path.exists(logpath):
    os.mkdir(logpath)
logging.basicConfig(filename=os.path.join(logpath, 'translate.log'), level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', )
logger = logging.getLogger()
"""
baidutranslation api
"""
myurl = "http://api.fanyi.baidu.com/api/trans/vip/translate"


def get_keyparams(q='hello world, i love you.'):
    appid = '20180302000130285'
    secretKey = 'lbbi4oZlxrCaWORBqpbt'
    fromLang = 'en'
    toLang = 'zh'
    salt = str(random.randint(32768, 65536))

    def createSIGN(q):
        sign = appid + q + str(salt) + secretKey
        m1 = md5()
        m1.update(sign.encode("utf8"))
        sign = m1.hexdigest()
        return sign

    keyparams = {'appid': appid,
                 'q': q,
                 'from': fromLang,
                 'to': toLang,
                 'salt': salt,
                 'sign': createSIGN(q)
                 }
    return keyparams


def _translateEN2ZHbyApiFrombaidu(En, reCount=5):
    keyparams = get_keyparams()
    trans = None
    for step in range(reCount):
        try:
            r = requests.get(myurl, params=keyparams)
        except Exception as  e:
            print(e)
        else:
            respone = r.json()
            #            print(respone)
            res = respone.get("trans_result")
            if res:
                trans = res[0]["dst"]
                break
            else:
                print(respone)
                logger.error(str(respone))
    return trans


"""
translate by  scrapy
"""
User_Agent = [
    "MQQBrowser/26 Mozilla/5.0 (Linux; U; Android 2.3.7; zh-cn; MB200 Build/GRJ22; CyanogenMod-7) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
    "JUC (Linux; U; 2.3.7; zh-cn; MB200; 320*480) UCWEB7.9.3.103/139/999",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:7.0a1) Gecko/20110623 Firefox/7.0a1 Fennec/7.0a1",
    "Opera/9.80 (Android 2.3.4; Linux; Opera Mobi/build-1107180945; U; en-GB) Presto/2.8.149 Version/11.10",
    "Mozilla/5.0 (Linux; U; Android 3.0; en-us; Xoom Build/HRI39) AppleWebKit/534.13 (KHTML, like Gecko) Version/4.0 Safari/534.13",
    "Mozilla/5.0 (iPhone; U; CPU iPhone OS 3_0 like Mac OS X; en-us) AppleWebKit/420.1 (KHTML, like Gecko) Version/3.0 Mobile/1A542a Safari/419.3",
    "Mozilla/5.0 (iPhone; U; CPU iPhone OS 4_0 like Mac OS X; en-us) AppleWebKit/532.9 (KHTML, like Gecko) Version/4.0.5 Mobile/8A293 Safari/6531.22.7",
    "Mozilla/5.0 (iPad; U; CPU OS 3_2 like Mac OS X; en-us) AppleWebKit/531.21.10 (KHTML, like Gecko) Version/4.0.4 Mobile/7B334b Safari/531.21.10",
    "Mozilla/5.0 (BlackBerry; U; BlackBerry 9800; en) AppleWebKit/534.1+ (KHTML, like Gecko) Version/6.0.0.337 Mobile Safari/534.1+",
    "Mozilla/5.0 (hp-tablet; Linux; hpwOS/3.0.0; U; en-US) AppleWebKit/534.6 (KHTML, like Gecko) wOSBrowser/233.70 Safari/534.6 TouchPad/1.0",
    "Mozilla/5.0 (SymbianOS/9.4; Series60/5.0 NokiaN97-1/20.0.019; Profile/MIDP-2.1 Configuration/CLDC-1.1) AppleWebKit/525 (KHTML, like Gecko) BrowserNG/7.1.18124",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows Phone OS 7.5; Trident/5.0; IEMobile/9.0; HTC; Titan)",
    "Mozilla/5.0 (Linux; Android 5.1.1; Nexus 6 Build/LYZ28E) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Mobile Safari/537.36",
]
headers = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 5.1.1; Nexus 6 Build/LYZ28E) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Mobile Safari/537.36"}

query_string = """
hello world.
"""
post_data = {
    "query": query_string.replace('\n', ' '),
    "from": "en",
    "to": "zh",
}


def getHEADERbyrandomUserAgent():
    headers["User-Agent"] = random.choice(User_Agent)
    return headers


post_url = "http://fanyi.baidu.com/basetrans"

from retry import retry

CrawlerRetry = -1


@retry(tries=CrawlerRetry, delay=0.1)
def _translateEN2ZHbyCrawlerFrombaidu(En):
    post_data["query"] = En.replace('\n', ' ')
    r = requests.post(post_url, data=post_data, headers=getHEADERbyrandomUserAgent())
    rjs = r.json()
    flag = rjs.get('errno')
    if flag is None or flag != 0:
        raise ValueError("errno is not 0")
    else:
        print("completed !!!")
        return rjs["trans"][0]["dst"]


def translateEN2ZH(En, reCount=5, mode='Crawler'):
    if mode == 'Crawler':
        trans = _translateEN2ZHbyCrawlerFrombaidu(En)
    elif mode == 'Api':
        trans = _translateEN2ZHbyApiFrombaidu(En, reCount)
    else:
        print("no implement for %s" % mode)
        trans = None
    return trans


import json

if __name__ == "__main__":
    """
    corpus loads
    """
    sourcePath = os.path.join(os.path.dirname(os.getcwd()), 'corpus')
    sourcePathList = ["dev_v1.1.json"]
    # ["dev_v1.1.json","dev_v2.0.json","eval_v2.0.json","test_public_v1.1.json","train_v1.1.json","train_v2.0.json"]

    FLAG_STOPREADLINES = False
    for sourceName in sourcePathList:
        rPath = os.path.join(sourcePath, sourceName)
        wPath = os.path.join(sourcePath, 'Trans2ZN_' + sourceName)
        PathLinesIndex = os.path.join(sourcePath, "LinesIndex_" + sourceName)
        if not os.path.exists(PathLinesIndex):
            LinesIndex_continue = 0
        else:
            with open(PathLinesIndex, 'r', encoding='utf8') as indf:
                LinesIndex_continue = int(indf.readline())
        print(rPath, '\n', wPath)
        try:
            with open(rPath, 'r', encoding="utf8") as fread:
                with open(wPath, 'a', encoding="utf8") as fwrite:
                    for linesIndex, streamF in enumerate(fread):
                        if linesIndex < LinesIndex_continue:
                            continue
                        print("opt lines {0:d}".format(linesIndex))
                        jsSingle = json.loads(streamF)
                        # answers
                        trans_answers = []
                        for answ in jsSingle["answers"]:
                            trans2ZH = translateEN2ZH(answ)
                            if trans2ZH:
                                trans_answers.append(trans2ZH)
                            else:
                                logger.info("line {0:d} can't translate keywords(answers)".format(linesIndex))
                                FLAG_STOPREADLINES = True

                        # query
                        trans_query = None
                        trans2ZH = translateEN2ZH(jsSingle["query"])
                        if trans2ZH:
                            trans_query = trans2ZH
                        else:
                            logger.info("line {0:d} can't translate keywords(query)".format(linesIndex))
                            FLAG_STOPREADLINES = True

                        # passages
                        trans_passageList = []
                        for passage in jsSingle["passages"]:
                            passage_text = None
                            trans2ZH = translateEN2ZH(passage.get("passage_text"))
                            if trans2ZH:
                                passage_text = trans2ZH
                            else:
                                logger.info("line {0:d} can't translate keywords(passage)".format(linesIndex))
                                FLAG_STOPREADLINES = True

                            passage["passage_text"] = passage_text
                            trans_passageList.append(passage)

                        jsSingle["passages"] = trans_passageList
                        jsSingle["query"] = trans_query
                        jsSingle["answers"] = trans_answers

                        # before writting,check
                        if FLAG_STOPREADLINES:
                            logger.info("fail to write msg on lines {0:d}".format(linesIndex))
                            break
                        else:
                            fwrite.write(json.dumps(jsSingle, ensure_ascii=False) + '\n')

        finally:
            with open(PathLinesIndex, 'w', encoding='utf8') as indf:
                indf.writelines(str(linesIndex))
