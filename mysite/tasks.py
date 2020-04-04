
 
import logging
 
from django.urls import reverse
from mysite.celery import app
import socket
import sys
 
@app.task
def get_ner_result(input):
    
    '''
    这里调tf模型求输出
    input是字典:{'model':模型名, 'txt': 输入文本}
    输出格式见下
    '''
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = socket.gethostname()
    port = 9999
    restext = ''
    #print(input)
    if(input['model'] == 'english'):
    	restext='''When <mark data-entity="person">Sebastian Thrun</mark> 
    	started working on self-driving cars at <mark data-entity="org">Google</mark> 
    	in <mark data-entity="date">2007</mark>, few people outside of the company took him seriously. 
    	“I can tell you very senior CEOs of major <mark data-entity="norp">American</mark> car companies 
    	would shake my hand and turn away because I wasn’t worth talking to,” said <mark data-entity="person">
    	Thrun</mark>, now the co-founder and CEO of online higher education startup Udacity, 
    	in an interview with <mark data-entity="org">Recode</mark><mark data-entity="date">earlier this week
    	</mark>.<br><br>A little <mark data-entity="date">less than a decade later</mark>, 
    	dozens of self-driving startups have cropped up while automakers around the world clamor, 
    	wallet in hand, to secure their place in the fast-moving world of fully automated transportation.'''
    	pass
    elif(input['model'] == 'chinese'):
        s.connect((host,port))
        s.send(str(input['txt']).encode('utf-8'))
        restext = s.recv(65536).decode('utf-8')
        s.close()
    return {'restext':restext}
