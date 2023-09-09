# Q1 여러 가지 활성화 함수의 forward 계층과 backward 계층 구현하기
import numpy as np
from common.functions import *
#from common.utils import *
from collections import OrderedDict
import torch

# 3) 지금부터 layers.py에 Sigmoid, Relu, SoftmaxwithLoss의 forward 연산과 backward 연산을 구현해 보려고 합니다. 아래 클래스와 함수의 원형을 참고하여 나머지를 완성하세요.
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = 1 / (1+np.exp(-x))
        self.out = out
        return out
    
    def backward(self,dout):
        dx = dout*(1.0-self.out)*self.out
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None #손실
        self.y = None #softmax의 출력
        self.t = None #정답 레이블(원-핫 벡터)

    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss
    
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
    
# 4) 이제 Affine 계층의 forward 연산과 backward 연산을 구현해 보려고 합니다. 아래 클래스와 함수의 원형을 참고하여 나머지를 layers.py에 완성하세요.
class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W)+self.b
        return out
    
    def backward(self,dout):
        dx = np.dot(dout,self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout,axis=0)
        return dx
