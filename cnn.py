from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.src.applications import VGG19
from keras.src.layers import BatchNormalization

train_dir = ""

generator = ImageDataGenerator()

train_ds = generator.flow_from_directory(train_dir, target_size=(128, 128), batch_size=16)

vgg19_net = VGG19(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

model_vgg19 = Sequential()
model_vgg19.add(vgg19_net)

model_vgg19.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform'))
model_vgg19.add(BatchNormalization())
model_vgg19.add(Dropout(0.2))

model_vgg19.add(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_uniform'))
model_vgg19.add(BatchNormalization())
model_vgg19.add(Dropout(0.2))

model_vgg19.add(Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_uniform'))
model_vgg19.add(BatchNormalization())
model_vgg19.add(Dropout(0.2))
model_vgg19.add(MaxPooling2D())

model_vgg19.add(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_uniform'))
model_vgg19.add(BatchNormalization())
model_vgg19.add(Dropout(0.2))
model_vgg19.add(MaxPooling2D())

model_vgg19.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform'))
model_vgg19.add(BatchNormalization())
model_vgg19.add(Dropout(0.2))

model_vgg19.add(Flatten())
model_vgg19.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model_vgg19.add(Dense(1, activation='sigmoid'))

model_vgg19.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_vgg19.summary()

history = model_vgg19.fit(train_ds, epochs=5, batch_size=16)
classes = list(train_ds.class_indices.keys())

loss, accuracy = model_vgg19.evaluate(train_ds, verbose=2)
print(accuracy)


