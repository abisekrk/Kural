# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()

# step - 1 - Convolution
classifier.add(Convolution2D(32, 3, 3 ,input_shape=(64, 64, 3), activation='relu'))

# step -2 -- Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# step - 3 -- Flattening
classifier.add(Flatten())

# Step -4 Full Connection
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=29, activation='sigmoid'))

# Compiling the CNN
classifier.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# part - 2 -- Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen= ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'asl_alphabet/asl_alphabet_train', # path/to/data/
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

print(training_set)

test_set = test_datagen.flow_from_directory(
    'asl_alphabet/asl_alphabet_test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

"""
from keras.utils.np_utils import to_categorical

nb_train_samples = len(training_set.filenames)
num_classes = len(training_set.class_indices)
train_labels = training_set.classes
#train_labels = to_categorical(train_labels, num_classes=num_classes)



nb_test_samples = len(test_set.filenames)
num_classes = len(test_set.class_indices)
test_labels = test_set.classes
#test_labels = to_categorical(test_labels, num_classes=num_classes)

print(test_labels)
"""


from keras.models import load_model



classifier.fit_generator(
    training_set,
    #samples_per_epoch=69600,
    steps_per_epoch=2175,
    epochs=25,
    validation_data=test_set,
    validation_steps=544
)

#Now Saving the model!!



classifier.save('projectKmodel2ver.h5')