from keras import datasets, models, layers, constraints, utils, optimizers, callbacks
from sklearn import model_selection

Sequential = models.Sequential
Dense = layers.Dense
Dropout = layers.Dropout
Flatten = layers.Flatten
Conv2D = layers.Conv2D
MaxPooling2D = layers.MaxPooling2D
max_norm = constraints.max_norm
to_categorical = utils.to_categorical
SGD = optimizers.SGD
ReduceLROnPlateau = callbacks.ReduceLROnPlateau

GridSearchCV = model_selection.GridSearchCV

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

model = Sequential()
model.add(Conv2D(28, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu', kernel_constraint=max_norm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(28, (3, 3), activation='relu', padding='same', kernel_constraint=max_norm(3)))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(28, (3, 3), activation='relu', padding='same', kernel_constraint=max_norm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

param_grid = dict(learning_rate=[.001, .01, .05, .1], batch_size=[10, 20, 30, 40, 50], epochs=[1, 2, 3, 4, 5])
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='balanced_accuracy')
grid_result = grid.fit(x_train, y_train)

epochs = grid_result.epochs
lrate = grid_result.learning_rate
decay = lrate/epochs
batch_size = grid_result.batch_size

# epochs = 5
# lrate = 0.01
# decay = lrate/epochs

sgd = optimizers.SGD(learning_rate=lrate, momentum=0.9, decay=decay, nesterov=False)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[reduce_lr])
scores = model.evaluate(x_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))
print(history.history)

model.save('image.keras')
