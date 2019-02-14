import numpy as np

from pytorch_pretrained_bert import BertTokenizer
from bert_serving.client import BertClient

bc = BertClient()
tokenizer = BertTokenizer.from_pretrained('./bert/vocab.txt')

def get_hiddens(sentences, lengths=None, end_mark=False):
    '''
    sentences: list of string
    '''
    states = bc.encode(sentences)
    _states = []
    for i, (sentence, state) in enumerate(zip(sentences, states)):
        if lengths is not None:
            _states.append(avg_hidden(sentence, state, lengths[i], end_mark[i]))
        else:
            _states.append(avg_hidden(sentence, state, None, end_mark[i]))
    return _states

def avg_hidden(sentence, state, length=None, end_mark=False):
    '''
    state: length * hidden_size
    '''
    _sentence = sentence
    if end_mark:
        begin_loc = 0
        sentence = [1] + [len(tokenizer.tokenize(x)) for x in sentence.split(' ') if x != ''] + [1]
        _state = np.zeros_like(state)
        if length is not None:
            # print(length)
            # print(len(sentence))
            assert length == len(sentence)
    else:
        begin_loc = 1 # 0 is CLS
        sentence = [len(tokenizer.tokenize(x)) for x in sentence.split(' ') if x != '']
        _state = np.zeros_like(state)[:len(sentence)]
    for i, num in enumerate(sentence):
        # if num == 0:
        #     print(i)
        #     print(_sentence)
        #     print(sentence)
        #     input()
        if i >= state.shape[0]:
            print('warning: sentence is too long')
            break
        if begin_loc >= state.shape[0]:
            print('warning: BPE is too long')
            _state[i] = state[-1]
        elif num == 1:
            _state[i] = state[begin_loc]
        elif num == 0:
            _state[i] = state[begin_loc]
        else:
            _state[i] = state[begin_loc:begin_loc + num].mean(axis=0)
        begin_loc += num
    return _state
