import numpy as np

# 의견: b의 값에 -를 붙이는게 아니라, tmp에서 +b대신 +(-b)로 표현하는게 직관적이지 않나?
def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.6,0.4]) #각 입력값의 영향력
    b = -0.6 #theta가 -b로 치환 (입력값이 얼마나 쉽게 활성화 되느냐)
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1
    # (b의 영향이 없기 위해서는) w가 큰값보다 -b가 같거나 커야 함

def NAND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([-0.6,-0.4])
    b = -(-0.61)
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1
    # (b의 영향이 없기 위해서는) w가 작은값(절대값이 큰값)보다 -b가 작아야(절대값이 커야)함

def OR(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.3,0.3])
    b = -0.29
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1
    # w의 영향을 크게 받음(b의 영향이 없는 답이 나오기 위해서는 가중치를 동등하게, -b를 가중치보다 작게 해야함)

def XOR(x1, x2): #한계: 여전히 비선형 데이터에 대해서 분류하기 어렵다.
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y

def step_function(x): #배열형태로 처리 가능
    y = x>0
    return y.astype(np.int) #true는 1로, false는 0으로

def sigmoid(x):
    return 1/(1+np.exp(-x)) #x값이 커질수록 1에 수렴(np.exp(-x)값이 작아지기 때문), 음수방향으로 커질수록 0에 수렴

def relu(x):
    return np.maximum(0,x)

def identity_function(x): #identify -> identity
    return x

def hyperbolic_tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def leaky_relu(x):
    negative = x*0.01
    return np.maximum(negative,x)

# 교재 내용
"""
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) # 오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y
"""

def softmax(x):
    ## 2차원인 경우와 1차원인 경우로 나누어서 계산
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis = 0)
        y = np.exp(x) / np.sum(np.exp(x), axis = 0)
        return y.T
    
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def sum_squares_error(y, t): #정답레이블뿐만 아니라, 오답레이블의 수치도 반영(원핫인코딩)
    return (0.5) * np.sum((y-t)**2)

def cross_entropy_error(y, t): #이 구현의 경우 원핫인코딩에서 t가 0인 원소는 교차엔트로피 오차도 0이므로 그 계산을 무시한다는 것이 핵심. 즉 정답에 해당하는 신경망의 출력만으로 교차 엔트로피 오차를 계산!
    if y.dim==1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arrange(batch_size), t] + delta)) / batch_size 
