import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from PIL import Image
from tensorflow.keras.preprocessing import image


pd.set_option('mode.chained_assignment', None)

train_data = pd.read_csv("C:/Users/henri/OneDrive/Área de Trabalho/projetos/TrafficsignData/Train.csv")
train_data['ClassId'] = train_data['ClassId'].astype(str)
for i in range(0, len(train_data['ClassId'])):
    if len(train_data['ClassId'][i]) == 1:
        train_data['ClassId'][i] = '0' + train_data['ClassId'][i]


test_data = pd.read_csv("C:/Users/henri/OneDrive/Área de Trabalho/projetos/TrafficsignData/Test.csv")
test_data['ClassId'] = test_data['ClassId'].astype(str)
for i in range(0, len(test_data['ClassId'])):
    if len(test_data['ClassId'][i]) == 1:
        test_data['ClassId'][i] = '0' + test_data['ClassId'][i]

img = Image.open('C:/Users/henri/OneDrive/Área de Trabalho/projetos/TrafficsignData/' + train_data['Path'][2])

pre_train = image.ImageDataGenerator(rescale=1./255, shear_range=0.2)
pre_test = image.ImageDataGenerator(rescale=1./255)

gen_train = pre_train.flow_from_dataframe(
    dataframe=train_data, directory='C:/Users/henri/OneDrive/Área de Trabalho/projetos/TrafficsignData/', x_col='Path',
    y_col='ClassId', target_size=(32, 32), batch_size=128, class_mode='categorical'
)

gen_test = pre_test.flow_from_dataframe(
    dataframe=test_data, directory='C:/Users/henri/OneDrive/Área de Trabalho/projetos/TrafficsignData/', x_col='Path',
    y_col='ClassId', target_size=(32, 32), batch_size=16, class_mode='categorical')

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation=tf.nn.relu))
model.add(Dense(43, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

model.fit(gen_train, verbose=1, epochs=15)

model.save('model_trained')
