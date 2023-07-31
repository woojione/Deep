import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import load_model

# Fashion MNIST 데이터 가져오기
(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

#모델 load
model=load_model('fashion_mnist_model.hdf5')


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# 모델 출력
model.evaluate(val_scaled, val_target)
a=int(input('원하는 이미지 번호: '))

#입력한 숫자에 대한 이미지 보여줌
plt.imshow(val_scaled[a].reshape(28, 28), cmap='gray_r')
plt.show()

# 예측률
preds = model.predict(val_scaled[a:a+1])
print('<각 카테고리 예측 확률>')
print(preds)

# 결과 예측
print('predict: '+class_names[np.argmax(preds)])
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0
model.evaluate(test_scaled, test_target)

#test 성능평가
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0
score = model.evaluate(test_scaled, test_target, verbose=0) # test 값 결과 확인
print('Test loss:', score[0])
print('Test accuracy:', score[1])

