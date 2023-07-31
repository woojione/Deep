from tensorflow import keras
from sklearn.model_selection import train_test_split

#fashion mnist data set 가져오기
(train_input,train_target),(test_input,test_target)=keras.datasets.fashion_mnist.load_data()

#픽셀값 정규화
train_scaled=train_input/255.0 #각 픽셀은 0~255 사이의 정수값을 가짐. 따라서 255로 나누어 0~1 사이의 값으로 정규화함.

#n차원 배열을 n-1차원 배열로 바꿈
train_scaled=train_scaled.reshape(-1,28*28) #28*28 이미지 크기에 맞게 1차원으로 변형

#훈련셋과 검증셋으로 나눔 (검증셋 20%)
train_scaled,val_scaled,train_target,val_target=train_test_split(train_scaled,train_target,test_size=0.2,random_state=42)

#층을 추가하는 법1
##<hidden layer 은닉층 2개 생성>
##dense1=keras.layers.Dense(100,activation='sigmoid',input_shape=(784,))
##dense2=keras.layers.Dense(10,activation='softmax')
##<model 생성>
##model=keras.Sequential([dense1,dense2]) #여러개의 층 추가하려면 은닉층을 리스트로 만들어 전달(*가장 처음 등장하는 은닉층부터 마지막까지 "순서대로" 나열해야함!)
##<층에 대한 정보 출력>
##model.summary()

#층을 추가하는 법2 (name은 써도 되고 안써도 됨)
#model=keras.Sequential([keras.layers.Dense(100,activation='sigmoid',input_shape=(784,),name='hidden'),keras.layers.Dense(10,activation='softmax',name='ouput')],name='fashion MNIST model')
#model.summary()

#층을 추가하는 법3
model=keras.Sequential()
model.add(keras.layers.Dense(100,activation='sigmoid',input_shape=(784,),name='hidden'))
model.add(keras.layers.Dense(10,activation='softmax',name='out'))
#model.summary()

#모델실행
model.compile(loss='sparse_categorical_crossentropy',metrics='accuracy')
#훈련
model.fit(train_scaled,train_target,epochs=5)

#검증
model.evaluate(val_scaled,val_target)