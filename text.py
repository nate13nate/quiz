from keras import datasets, models, layers, optimizers, callbacks
from sklearn import model_selection
import tensorflow

ReduceLROnPlateau = callbacks.ReduceLROnPlateau
GridSearchCV = model_selection.GridSearchCV
Tokenizer = tensorflow.keras.preprocessing.text.Tokenizer

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data()

tokenizer = Tokenizer(num_words=2000)

model = models.Sequential()
model.add(layers.LSTM(30))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(30))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(30))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='softmax'))

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
history = model.fit((x_train), (y_train), validation_data=((x_test), (y_test)), epochs=epochs, batch_size=batch_size, callbacks=[reduce_lr])
scores = model.evaluate(x_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))
print(history.history)

model.save('text.keras')
