import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical # one-hot 인코딩
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Fashion MNIST 데이터 가져오기
(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

# 합성곱 신경망 만들기
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', 
                              padding='same', input_shape=(28,28,1)))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', 
                              padding='same'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax')) #확률 중 가장 높은 애를 고르기 위해 softmax사용
model.summary()


# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics='accuracy')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics='accuracy')


#모델 훈련
history = model.fit(train_scaled, train_target, epochs=20,
                    validation_data=(val_scaled, val_target))

#모델 저장
model.save('fashion_mnist_model.hdf5')

