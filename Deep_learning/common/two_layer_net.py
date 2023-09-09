import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    # a. 해당 함수는 가중치를 초기화 하는 함수입니다. Input_size는 입력층의 노드, hidden_size는 은닉층의 노드, output_size는 출력층의 노드 수 입니다. 이를 참고한 후 딕셔너리를 이용하여 가중치를 초기화하는 함수를 구현하세요. (해당 함수는 클래스에서 초기화가 일어나는 부분이기 때문에 리턴값이 없습니다.)
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)
    
    # b. 해당 함수는 초기화된 가중치를 이용하여 신경망의 연산 과정이 일어나는 부분입니다. 지난 주와 동일한 방법으로 결과를 리턴하도록 구현하세요. 리턴해야 하는 것은 y(예측값) 입니다.
    def predict(self, x):
        W1,W2 = self.params['W1'], self.params['W2']
        b1,b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x,W1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2)+b2
        y = softmax(a2)
        return y
    
    # c. 해당 함수는 예측 결과와 실제 정답 간의 정확도를 계산하는 부분입니다. X와 t는 각각 입력 데이터와 정답 레이블입니다. 이를 참고하여 정확도 accuracy를 계산하고 리턴하세요.
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)
        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy

    # d. 해당 함수는 예측 결과와 실제 정답 간의 손실을 계산하는 부분입니다. 계산 과정을 구현하고 loss를 리턴하세요. Loss function은 자율적으로 선택하시고, 해당 loss function을 선택한 이유를 설명하세요.
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y,t)

    # e. 해당 함수는 계산한 loss를 바탕으로 가중치를 업데이트하는 함수입니다. 딕셔너리를 이용하여 업데이트된 가중치를 리턴하도록 구현하세요.
    def numerical_gradient(self, x, t):
        loss_W = lambda W:self.loss(x,t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads