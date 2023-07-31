import tensorflow as tf
mnist = tf.keras.datasets.mnist

#혹시 작동 안되면 ctrl+shift+p 해서 인터프리터가 tensorflow로 되어있는지 확인!
#conda activate tensorflow로 아나콘다 가상환경 활성화 시켜줘야함.거기에 tensorflow랑 keras 깔아놨음.

(x_train, y_train),(x_test, y_test) = mnist.load_data() #이미 작성된 데이터베이스인 mnist에서 데이터를 불러와서 변수에 넣어줌
x_train, x_test = x_train / 255.0, x_test / 255.0 #기존 int형태를 float 형태로 바꿔줌. x_train: 학습을 위한 데이터, x_test:학습된 데이터를 검증하는 검증 데이터


#sequential(순차적)방식으로 모델 만들어줌
model = tf.keras.models.Sequential([ 
  tf.keras.layers.Flatten(input_shape=(28, 28)), #28x28 형태로 받은걸 1차원 구조로 평탄하게 바꿔줌
  tf.keras.layers.Dense(128, activation='relu'), #layer층 추가해 밀도를 높게 해줌. 128개의 입력을 받고 활성함수로 렐루함수 사용
  tf.keras.layers.Dropout(0.2), #input받은 데이터에서 버릴 유닛의 수를 정하는 함수. 0~1(0~100%)을 넣어 drop시킴. 0.2->20% drop
  tf.keras.layers.Dense(10, activation='softmax') #다시 layer 추가. 활성화 함수로 소프트맥스함수(=소프트맥스회귀) 사용
])

#모델 실행
model.compile(optimizer='adam',#최적화(손실값 최소화)-손실함수를 통해 얻은 손실값으로부터 모델을 업데이트함
              loss='sparse_categorical_crossentropy', #손실함수=희소(정수로된 타깃값만 사용) 다중분류, 타깃값을 원-핫 인코딩으로 사용시 loss='categorical_crossentropy' 
              metrics=['accuracy']) #평가기준=accuracy(정확도 지표)

#fit: 모델 훈련 메소드(훈련셋)
model.fit(x_train, y_train, epochs=5) #x학습데이터,y학습데이터, epochs:총 5번 학습

#evaluate: 모델의 성능을 평가하는 메소드(검증셋)
model.evaluate(x_test, y_test)