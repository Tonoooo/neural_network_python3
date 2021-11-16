import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(latih_gambar, latih_label), (uji_gambar, uji_label) = data.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
latih_gambar = latih_gambar/255.0
uji_gambar = uji_gambar/255.0
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dense(10,activation="softmax")
    ])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(latih_gambar,latih_label,epochs=5)
test_loss, test_acc = model.evaluate(uji_gambar, uji_label)
print('\nMenguji akurasinya: ', test_acc)

## Memprediksi
memprediksi = model.predict(uji_gambar) # jadi model kita akan memprediksi apa yang ada di uji_gambar
## Menampilkan prediksi
plt.figure(figsize=(5,5))
for i in range(5): # 0 sampai 4, akan mentest dan menunjukan 5 gambar
    plt.grid(False)
    plt.imshow(uji_gambar[i], cmap=plt.cm.binary) # cmap=plt.cm.binary agar warnanya hitam putih
    plt.xlabel("test :"+class_names[uji_label[i]])
    plt.title("hasil prediksinya:"+class_names[np.argmax(memprediksi[i])])
    plt.show()
    # cara kerjanya:
    # pertama: memprediksi[i] => model akan memprediksi apa itu index ke (berapapun itu yang diberikan oleh si uji)
    # kedua: setelah model mengeluarkan output, kan ada 10 neuron yang memberi output, ini(np.argmax) akan
    #    mengambil nilai tertinggi dan akan menemukan index itu
    # ketiga: setelah indexnya ketemu maka dicocokanlah dengan ini(class_names)
    # keempat: keluarlah