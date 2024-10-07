# day27 -> 1_케라스.py  # p.50

import numpy as np
import matplotlib.pyplot as plt
# [1]
def make_linear( w = 0.5 , b = 0.8 , size = 50 , noise = 1.0 ) :
    x = np.random.rand( size ) # 0 ~ 1 사이의 size 만큼의 난수로 이루어진 x배열 선언
    print( x )  # [ 0.43874181 ~~~  ]
    y = w * x + b # 0.63162254 = 0.3 * 0.43874181 + 0.5
    print( y )
    # 음수 noise 부터 양수 noise 사이의 난수 생성 # 개수는 y개수만큼 # 크기는 y의 크기만큼
    # 실제Y값의 노이즈(작은변화) 를 주고 확인 하는 예제
    noise = np.random.uniform( -abs(noise) , abs(noise) , size = y.shape )
    print( noise )
    yy = y + noise
    # 시각화
    plt.plot( x , y , c = 'r'  ) # 선차트
    plt.scatter(x, yy) # 선점도 차트
    plt.title( 'y = 0.3 * x + 0.5')
    plt.show()
    return  x , yy
x , y = make_linear( w = 0.3 , b = 0.5 , size=100 , noise=0.01 )

# [2] w(기울기) 와 b(y절편)를 학습률(업데이트)하며 손실(오차)를 최소화 하는 방법를 찾는 예제
num_epoch = 1000 # 1. 학습 횟수는 최대 1000번
learning_rate = 0.005 # 2.업데이트 크기 # 학습률
errors = [ ] # 3. 에러 기록 # 손실(오차)를 기록할 리스트

w = np.random.uniform( low=0.0 , high=1.0 ) #  w :기울기을 0 과 1 사이의 랜덤 값 으로 초기화
print( w )
b = np.random.uniform( low=0.0 , high=1.0) #  b :y절편을 0 과 1 사이의 랜덤 값 으로 초기화
print( b )

for epoch in range( num_epoch ) : # 최대 에포크 수만큼 반복문
    y_hat = w * x + b # 예측값 # 주어진 x에 따른 최적의 오차를 갖는  w와 b 찾기
    # 오차 계산식 # 평균 제곱 오차 계산식
        # 예측값 과 실제값( 노이즈된 y값) 차이를 제곱하여 평균한 값
    error = 0.5 * ( ( y_hat - y ) ** 2 ).sum()
    # 만약에 오차가 0.004 이면 반복문 종료
    if error < 0.005 :
        break
    # 기울기 미분 계산
        # 1. 기울기 업데이트
    w -= learning_rate * ( (y_hat-y) * x ).sum()
        # 2. y절편 업데이트
    b -= learning_rate * ( y_hat - y ).sum()
    # 손실(오차) 기록
    errors.append( error )
    print( f'에포크/학습수 : { epoch } , 기울기:{ w:.1f} , y절편:{ b:.1f} , 오차/실제값과차이 : { error:.5f}' )

plt.plot( errors )
plt.xlabel('epochs')
plt.ylabel('error')
plt.show()

















