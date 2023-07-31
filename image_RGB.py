import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


#training data set 만들고(r,g,b) -> 넣어서 훈련시켜서 -> 진행

# Load the Google logo image
img = cv2.imread('wm_r.jpeg')

# Preprocess the image data
data = img / 255.0
data = data.reshape(-1, 3)

# Define the model
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', input_shape=(1, 1, 3)))
model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

#손실함수 평균제곱오차 사용했음
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#mean_absolute_error : 15%
#hinge : 28%
#binary_cross_entropy : 28%
#categorical_crossentropy : 35%
#mean_squared_error : 58%

# Fit the model on the data
model.fit(data.reshape(-1, 1, 1, 3), data, epochs=10)

# Predict the RGB values for each pixel in the image
predictions = model.predict(data.reshape(-1, 1, 1, 3))

# Reshape the predictions to match the shape of the original image
predictions = predictions.reshape(img.shape)

# Calculate the percentage of R, G and B colors in the image
r_percent = np.sum(predictions[:,:,0]) / np.sum(predictions) * 100
g_percent = np.sum(predictions[:,:,1]) / np.sum(predictions) * 100
b_percent = np.sum(predictions[:,:,2]) / np.sum(predictions) * 100

# Print the percentage of R, G and B colors in the terminal window
print("R: {:.2f}%, G: {:.2f}%, B: {:.2f}%".format(r_percent, g_percent, b_percent))

# Display the resulting image
cv2.imshow('image', predictions)
cv2.waitKey(0)
cv2.destroyAllWindows()
