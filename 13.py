#задание 13
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#функция для создания модели 
def create_model(lr):
    initializer = tf.keras.initializers.HeNormal()  
    model = tf.keras.Sequential([

        tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer, input_shape=(100,)),
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=initializer)
    ])
    # компиляция модели
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#генерация данных (100 признаков и 10 классов)
X_train = np.random.randn(2000, 100).astype(np.float32)  
y_train = np.random.randint(0, 10, size=(2000,))         

X_val = np.random.randn(500, 100).astype(np.float32)     
y_val = np.random.randint(0, 10, size=(500,))           

#функция для обучения и сбора истории + норм градиентов
def train_and_collect(lr):
    model = create_model(lr) 
    grad_norms = []          
    losses = []               
    accs = []              

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)  # оптимизатор SGD
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()  # функция потерь

    for epoch in range(10):  
        #обучение по батчам
        for i in range(0, len(X_train), 64):  
            x_batch = X_train[i:i+64]       
            y_batch = y_train[i:i+64]         
            with tf.GradientTape() as tape:   
                logits = model(x_batch, training=True)  
                loss_value = loss_fn(y_batch, logits)   
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))  

            #считаем норму градиента
            total_norm = np.sqrt(sum([tf.reduce_sum(g**2).numpy() for g in grads if g is not None]))
            grad_norms.append(total_norm) 

        #валидация
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0) 
        losses.append(val_loss)
        accs.append(val_acc)  


    return losses, accs, grad_norms  # возвращаем историю обучения

#запуск для lr
results = {}
for lr in [0.001, 0.01, 0.1]:  
    losses, accs, grad_norms = train_and_collect(lr)
    results[lr] = {"losses": losses, "accs": accs, "grad_norms": grad_norms}

#построение графиков
plt.figure(figsize=(15,5))

# график функции потерь
plt.subplot(1,3,1)
for lr in results:
    plt.plot(results[lr]["losses"], label=f"lr={lr}")
plt.title("Loss")
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.legend()

# график точности
plt.subplot(1,3,2)
for lr in results:
    plt.plot(results[lr]["accs"], label=f"lr={lr}")
plt.title("Accuracy")
plt.xlabel("Эпоха")
plt.ylabel("Accuracy")
plt.legend()

# график норм градиентов
plt.subplot(1,3,3)
for lr in results:
    plt.plot(results[lr]["grad_norms"], label=f"lr={lr}")
plt.title("Градиенты")
plt.xlabel("Батчи")
plt.ylabel("grad")
plt.legend()
plt.tight_layout()
plt.show()

