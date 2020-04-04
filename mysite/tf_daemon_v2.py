# -*- coding: utf-8 -*-
import sys
import os
import socket
import tensorflow as tf
import numpy as np
import Weibo_model
import codecs
import re
from concurrent.futures import ThreadPoolExecutor
import threading
import time

rNUM = '(-|\+)?\d+((\.)\d+)?%?'
rENG = '[A-Za-z_.]+'
vector = []
word2id = {}
id2word = {}
tag_id = {}
id_tag={}
word_dim=100

def load_embedding(setting):
    print 'reading chinese word embedding.....'
    f = open('./data/embed.txt','r')
    f.readline()
    while True:
        content=f.readline()
        if content=='':
            break
        else:
            content=content.strip().split()
            word2id[content[0]]=len(word2id)
            id2word[len(id2word)]=content[0]
            content=content[1:]
            content=[float(i) for i in content]
            vector.append(content)
    f.close()
    word2id['padding']=len(word2id)
    word2id['unk']=len(word2id)
    vector.append(np.zeros(shape=setting.word_dim,dtype=np.float32))
    vector.append(np.random.normal(loc=0.0,scale=0.1,size=setting.word_dim))
    id2word[len(id2word)]='padding'
    id2word[len(id2word)]='unk'

def process_train_data(setting):
    print 'reading train data.....'
    train_word=[]
    train_label=[]
    train_length=[]
    f=open('./data/weiboNER.conll.train','r')
    train_word.append([])
    train_label.append([])
    train_max_len=0
    while True:
        content=f.readline()
        if content=='':
            break
        elif content=='\n':
            length=len(train_word[len(train_word)-1])
        else:
            content=content.replace('\n','').replace('\r','').strip().split()
            if content[1]!='O':
                label1=content[1].split('.')[0]
                label2=content[1].split('.')[1]
                content[1]=label1
                if label2=='NOM':
                    content[1]='O'
            if content[0] not in word2id:
                word2id[content[0]]=len(word2id)
                vector.append(np.random.normal(loc=0.0,scale=0.1,size=setting.word_dim))
                id2word[len(id2word)]=content[0]
            if content[1] not in tag_id:
                tag_id[content[1]]=len(tag_id)
                id_tag[len(id_tag)]=content[1]
            

def process_test_data(setting):
    print 'reading test data.....'
    test_word=[]
    test_label=[]
    test_length=[]
    f=open('./data/weiboNER.conll.test','r')
    test_word.append([])
    test_label.append([])
    test_max_len=0
    while True:
        content=f.readline()
        if content=='':
            break
        elif content=='\n':
            if len(test_word[len(test_word)-1])>test_max_len:
                test_max_len=len(test_word[len(test_word)-1])
            test_word.append([])
            test_label.append([])
        else:
            content = content.replace('\n', '').replace('\r', '').strip().split()
            if content[1]!='O':
                label1=content[1].split('.')[0]
                label2=content[1].split('.')[1]
                content[1]=label1
                if label2=='NOM':
                    content[1]='O'
            if content[0] not in word2id:
                word2id[content[0]]=len(word2id)
                vector.append(np.random.normal(loc=0.0,scale=0.1,size=setting.word_dim))
                id2word[len(id2word)]=content[0]
            if content[1] not in tag_id:
                tag_id[content[1]]=len(tag_id)
                id_tag[len(id_tag)]=content[1]
           

def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += unichr(inside_code)
    return rstring

def preprocess(filename):
    sentence=[]
    length=[]
    label=[]
    max_len=0
    with codecs.open(filename,'r','utf-8') as f:
        print 'reading cws data.....'
        for line in f:
            sent=strQ2B(line).split()
            new_sent=[]
            sent_label=[]
            for word in sent:
                word=re.sub(rNUM,'0',word)
                word=re.sub(rENG,'X',word)
                for i in range(len(word)):
                    if word[i] not in word2id:
                        word2id[word[i]]=len(word2id)
                        vector.append(np.random.normal(loc=0.0, scale=0.1, size=word_dim))
                        id2word[len(id2word)]=word[i]


def dec(logits,trans_params,lengths):
    viterbi_sequences=[]
    for logit, length in zip(logits, lengths):
        logit = logit[:length]
        viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
        viterbi_sequences += [viterbi_seq]
    return viterbi_sequences
                    
def worker_thread(_args):  
    msg,clientsocket, chn = _args
    words_batch = []
    length_batch = []
    for i in range((len(msg) + 79) // 80):
        words = []
        for j in range(min(80, len(msg) - 80*i)):
            #if msg[i*80+j] not in word2id:
                #word2id[msg[i*80+j]]=len(word2id)
                #tmp = np.random.normal(loc=0.0,scale=0.1,size=100)
                #chn.word_embed = np.column_stack((chn.word_embed, tmp))
                #id2word[len(id2word)]=msg[i*80+j]
            words.append(word2id[msg[i*80+j].encode('utf-8')])
        length_batch.append(np.asarray(len(words)))
        if(len(words) < 80):
            for _ in range(80 - len(words)):
                words.append(word2id['padding'])
        #print(words)
        words_batch.append(np.asarray(words))

    feed_dict = {}
    feed_dict[chn.input]= words_batch
    feed_dict[chn.sent_len]= length_batch
    feed_dict[chn.is_ner]=1
    logits, trans_params, at1, at2= chn_sess.run([chn.ner_project_logits,chn.ner_trans_params,
        chn.ner_attention_weight,chn.shared_attention_weight],feed_dict)
    at1 = np.divide(np.sum(at1, axis=(1,2)), 8*np.array(length_batch)[:,None])
    at2 = np.divide(np.sum(at2, axis=(1,2)), 8*np.array(length_batch)[:,None])
    at = np.divide(np.add(at1,at2), 2)
    print(at.shape)
    #print(at)
    viterbi_sequences=dec(logits,trans_params,length_batch)
    ans = []
    for i in range(len(viterbi_sequences)):
        ans.extend([id_tag[x] for x in viterbi_sequences[i]])
    print(ans)
    #restext = u'<mark data-entity="person">小明</mark>上个月去了<mark data-entity="org">联合国</mark>开会。'
    state = ''
    restext = ''
    for i in range(len(msg)):
        if(ans[i][0] == 'B'):
            if state != '':
                restext = restext + '</mark>'
            state = ans[i][2:]
            restext = restext + '<mark data-entity=' + state.lower() + '>'
            restext = restext + msg[i]
        elif(ans[i] == 'O'):
            if state != '':
                restext = restext + '</mark>'
            state = ''
            restext = restext + msg[i] + '(' + ('%.1f' % (at[i//80][i%80] * length_batch[i//80])) + ')'
        else:
            restext = restext + msg[i]
    if state != '':
        restext = restext + '</mark>'

    clientsocket.send(restext.encode('utf-8'))
    clientsocket.close()
    return restext


pool = ThreadPoolExecutor(max_workers=5)
setting=Weibo_model.Setting()
load_embedding(setting)
process_train_data(setting)
process_test_data(setting)

filename='./data/msr_training.utf8'
preprocess(filename)
#print(id2word)
#print(id_tag)

def get_result(future):
    print(future.result())

setting = Weibo_model.Setting()


with tf.Graph().as_default():
    config = tf.ConfigProto(allow_soft_placement=True)
    chn_sess=tf.Session(config=config)
    with chn_sess.as_default():
        embedding = np.array(vector)
        with tf.variable_scope('ner_model'):
            chn = Weibo_model.TransferModel(setting, tf.cast(embedding, tf.float32), adv=True, is_train=False)
            chn.multi_task()

        saver=tf.train.Saver()
        k=4440
        saver.restore(chn_sess, './ckpt/lstm+crf' + '-' + str(k))

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host = socket.gethostname()
        port = 9999
        s.bind((host, port))
        s.listen(10)
        while True:
            clientsocket,addr = s.accept()
            msg = clientsocket.recv(65536).decode('utf-8').replace(' ','').replace('\n','').replace('\r','').strip()
            #print(msg)
            future = pool.submit(worker_thread, [msg,clientsocket, chn])
            #print(future.done())
            future.add_done_callback(get_result)
        #pool.shutdown()
            
			
            
