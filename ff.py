import tensorflow as tf
import numpy as np

# Let's say you have three different models
def create_client_model1():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

def create_client_model2():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

def create_client_model3():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),  # Flatten the 28x28 images into a vector of 784
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])


# Dummy dataset (EMNIST) and common validation set
(X_train, y_train), (X_val, y_val) = tf.keras.datasets.mnist.load_data()
X_train, X_val = X_train / 255.0, X_val / 255.0
X_train = np.expand_dims(X_train, -1)
X_val = np.expand_dims(X_val, -1)

# Create each client model
client_model1 = create_client_model1()
client_model2 = create_client_model2()
client_model3 = create_client_model3()

# Each client trains its own model locally
client_model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
client_model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
client_model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

client_model1.fit(X_train, y_train, epochs=1)
client_model2.fit(X_train, y_train, epochs=1)
client_model3.fit(X_train, y_train, epochs=1)

# Each client sends predictions on a shared validation set
client_predictions1 = client_model1.predict(X_val)
client_predictions2 = client_model2.predict(X_val)
client_predictions3 = client_model3.predict(X_val)

# Aggregate predictions (e.g., by averaging)
aggregated_predictions = (client_predictions1 + client_predictions2 + client_predictions3) / 3

# The server uses aggregated predictions to train a global model (a new model)
global_model = create_client_model1()  # Global model can have any architecture
global_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train global model on aggregated predictions (using soft labels)
y_val_soft = tf.keras.utils.to_categorical(np.argmax(aggregated_predictions, axis=1), num_classes=10)
global_model.fit(X_val, y_val_soft, epochs=1)

