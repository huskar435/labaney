# задание 7
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#функция для создания модели
def create_model(initializer):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer, input_shape=(100,)),
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=initializer)
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# генерация случайных входных данных и меток 
X = np.random.randn(1000, 100).astype(np.float32)
y = tf.keras.utils.to_categorical(np.random.randint(0, 10, size=(1000,)), num_classes=10)

# сallback для статистики активаций
class ActivationStats(tf.keras.callbacks.Callback):
    def __init__(self, X):
        super().__init__()
        self.X = X

    def on_epoch_end(self, epoch, logs=None):
        activations = []
        inp = self.X[:200]  
        for layer in self.model.layers[:-1]: 
            inp = layer(inp)
            activations.append(inp.numpy())
        print(f"Epoch {epoch+1}:")
        for i, act in enumerate(activations):
            print(f" Слой {i+1}: mean={act.mean():.4f}, std={act.std():.4f}")

#создаем модели
model_he = create_model(tf.keras.initializers.HeNormal())
model_glorot = create_model(tf.keras.initializers.GlorotNormal())

#обучаем и выводим статистику
print("HeNormal")
model_he.fit(X, y, epochs=5, batch_size=32, verbose=0, callbacks=[ActivationStats(X)])

print("GlorotNormal")
model_glorot.fit(X, y, epochs=5, batch_size=32, verbose=0, callbacks=[ActivationStats(X)])