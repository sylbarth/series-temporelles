import tensorflow as tf

# Modèle LSTM sur données journalière, prenant un historique de 20 jours
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(20, 1)),
    tf.keras.layers.Dense(1)
])

# Compilation du modèle
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Prepare data
X_train = ... # input data with shape (batch_size, 20, 1)
y_train = ... # target data with shape (batch_size, 1)

X_test = ... # test data with shape (batch_size, 20, 1)
y_test = ... # target test data with shape (batch_size, 1)


# train du modèle sur 20 epochs sans validation
model.fit(X_train, y_train, epochs=20)

# train du modèle sur 20 epochs avec validation
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))


