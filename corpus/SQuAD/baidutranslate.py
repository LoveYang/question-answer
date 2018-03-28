# -*- coding: utf-8 -*-

import requests
import json
import os
import random

query_string = """We study automatic question generation
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

post_data = {
    "query": query_string.replace('\n', ' '),
    "from": "en",
    "to": "zh",
}


def getHEADERbyrandomUserAgent():
    headers["User-Agent"] = random.choice(User_Agent)
    return headers


num = 0
import logging

logger = logging.getLogger('mytest')
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
from retry import retry


@retry(tries=-1, delay=0.1)
def test(*arg):
    num = 0
    logger.info('pidid: ', num)
    r = requests.post(post_url, data=post_data, headers=getHEADERbyrandomUserAgent())
    try:
        rjs = r.json()
    except json.decoder.JSONDecodeError:
        print(r.content)
        print("json erro")
        raise json.decoder.JSONDecodeError

    else:
        flag = rjs.get('errno')
        if flag is None:
            print("not flag")
            print(rjs)

        elif flag != 0:
            print(rjs)
        else:
            print(rjs["trans"][0]["dst"])
            num = 1
            print('pidid is complete: ', num)

    return arg[0]


post_url = "http://fanyi.baidu.com/basetrans"
# for step in range(100):
#    r = requests.post(post_url,data=post_data,headers=getHEADERbyrandomUserAgent())
#    #print(r.json())
#    dict_ret = r.json()
#    ret = dict_ret["trans"][0]["dst"]
#    print("result is :",ret)
if __name__ == "__main__":
    import time

    curT = time.time()
    #    test()
    from multiprocessing import Pool

    numclass = 8
    pool = Pool(numclass)
    subtask = 30
    print(pool.map(test, [x for x in range(subtask)]))
    print('speed of task is : ', (time.time() - curT) / subtask)
    print('speed of process is : ', (time.time() - curT) / subtask / numclass)
