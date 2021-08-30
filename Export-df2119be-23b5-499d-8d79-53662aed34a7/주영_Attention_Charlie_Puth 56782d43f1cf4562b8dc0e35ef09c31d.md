# 주영_Attention_Charlie_Puth

### 16.1 어텐션 메커니즘 (Attention Mechanism)

RNN에 기반한 seq2seq 모델에는 크게 두 가지 문제가 있다.

(1) 하나의 고정된 크기의 벡터에 모든 정보를 압축하려고 하니까 정보 손실이 발생한다.

(2) RNN의 고질적인 문제인 기울기 소실(Vanishing Gradient) 문제가 존재한다.

### 16.1.1 어텐션(Attnetion)의 아이디어

디코더에서 출력 단어를 예측하는 매 시점(time step)마다 인코더에서의 전체 입력 문장을 다시 한 번 참고한다. 이 때, 예측해야할 단어와 연관이 있는 입력 단어 부분을 좀 더 집중(attention)해서 보게 된다.

### 16.1.2 어텐션 함수(Attention Function)

![쿼리.png](%E1%84%8C%E1%85%AE%E1%84%8B%E1%85%A7%E1%86%BC_Attention_Charlie_Puth%2056782d43f1cf4562b8dc0e35ef09c31d/%EC%BF%BC%EB%A6%AC.png)

$$Attention(Q,K,V) = Attention Value$$

이 때 각 변수는 다음을 나타낸다.

```jsx
Q = Query : t 시점의 디코더 셀에서의 은닉 상태
K = Keys : 모든 시점의 인코더 셀의 은닉 상태들
V = Values : 모든 시점의 인코더 셀의 은닉 상태들
```

### 16.1.3 닷-프로덕트 어텐션(Dot-Product Attention)

(1) 어텐션 스코어(Attention Score)를 구한다.

(2) 소프트맥스(softmax) 함수를 통해 어텐션 분포(Attention Distribution)를 구한다.

(3) 각 인코더의 어텐션 가중치와 은닉 상태를 가중합하여 어텐션 값(Attention Value)을 구한다.

(4) 어텐션 값과 디코더의 t 시점의 은닉 상태를 연결한다.(Concatenate)

(5) 출력층 연산의 입력이 되는 $s_t$를 계산한다.

(6) $s_t$를 출력층의 입력으로 사용한다.

### 16.1.4 다양한 종류의 어텐션 (Attention)

닷-프로덕트 어텐션과 다른 어텐션들의 차이는 어텐션 스코어 함수의 차이에 있다. 닷-프로덕트 어텐션은 어텐션 스코어를 구하는 방법이 내적이었고, 그 외에도 여러가지 방법이 존재한다.

### 16.2 바다나우 어텐션(Bahdanau Attention)

(1) 어텐션 스코어(Attention Score)를 구한다.

(2) 소프트맥스(softmax) 함수를 통해 어텐션 분포(Attention Distribution)를 구한다.

(3) 각 인코더의 어텐션 가중치와 은닉 상태를 가중합하여 어텐션 값(Attention Value)을 구한다.

(4) 컨텍스트 벡터로부터 $s_t$를 구한다.

### 16.3 양방향 LSTM과 어텐션 메커니즘(BiLSTM with Attention mechanism)