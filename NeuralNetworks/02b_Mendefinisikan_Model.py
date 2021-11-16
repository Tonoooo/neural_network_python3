# Text Classification

import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
imdb = keras.datasets.imdb
(latih_data, latih_label), (uji_data, uji_label) = imdb.load_data(num_words=10000)
kata_index = imdb.get_word_index()
kata_index = {k:(v+3) for k, v in kata_index.items()}
kata_index["<PAD>"] = 0
kata_index["<START>"] = 1
kata_index["<UNK>"] = 2
kata_index["<UNUSED>"] = 3
dibalik_kata_index = dict([(value,key) for (key,value) in kata_index.items()])
latih_data = keras.preprocessing.sequence.pad_sequences(latih_data, value=kata_index["<PAD>"], padding="post", maxlen=250)
uji_data= keras.preprocessing.sequence.pad_sequences(uji_data, value=kata_index["<PAD>"], padding="post", maxlen=250)
def decode_review(text):
    return " ".join([dibalik_kata_index.get(i, "?") for i in text])

## -------------- Mendefinisikan Modelnya -----------------------
model = keras.Sequential()
model.add(keras.layers.Embedding(10000,16)) # 10000 neuron dengan 16 dimensi
model.add(keras.layers.GlobalAveragePoolingID())
model.add(keras.layers.Dense(16, activation="relu")) # l6 neuron
model.add(keras.layers.Dense(1, activation="sigmoid")) # 1 output neuron
model.summary()
# .summary() = fungsi yang memberikan gambaran umum tentang koefisien model dan seberapa cocoknya,
# bersama dengan beberapa ukuran statistik lainnya. Seperti Rangkuman bersifat tekstual dan mencakup
# informasi tentang: Lapisan dan urutannya dalam model. Bentuk keluaran dari setiap lapisan, Nama dan jenis
# semua lapisan dalam model. Bentuk keluaran untuk setiap lapisan.
"""
Berikut adalah urutan lapisan yang kami definisikan untuk model klasifikasi teks kami:
- Lapisan Embedding/Penyematan Kata
- GlobalAveragePooling1D
- Dense/Padat
- Dense/Padat
Kita akrab dengan apa itu dua lapisan padat, tetapi apa sebenarnya lapisan penyematan dan apa yang 
dilakukan GlobalAveragePooling1D?

----- Embedding: -----
kita akan membandingkan dua kalimat yang sangat sederhana. Pertama dalam bentuk yang dapat dibaca manusia dan
kedua dalam bentuk yang disandikan bilangan bulat:
Dapat Dibaca Manusia:
Have a great day (Semoga harimu menyenangkan)
Have a good day (Semoga harimu menyenangkan)
Enkode bilangan bulat:
[0, 1, 2, 3]
[0, 1, 4, 3]
Pemetaan: {0: "Have", 1: "a", 2: "great", 3: "day", 4: "good"}
Melihat kalimat-kalimat di atas, kita sebagai manusia tahu bahwa kedua kalimat tersebut sangat mirip dan
memiliki arti yang sama. Namun ketika kita melihat versi pengkodean bilangan bulat yang dapat kita ketahui
adalah bahwa kata-kata pada indeks 2 (posisi 3) berbeda. Kami tidak tahu betapa berbedanya mereka.
Di sinilah lapisan embedding  kata masuk. Kami ingin cara untuk menentukan tidak hanya isi kalimat tetapi 
konteks kalimat. Lapisan embedding akan berusaha menentukan arti setiap kata dalam kalimat dengan
memetakan setiap kata ke posisi dalam ruang vektor. Jika Anda tidak peduli dengan matematika atau tidak 
memahaminya, anggap itu hanya mengelompokkan kata-kata yang mirip.

Contoh sesuatu yang kami harapkan akan dilakukan oleh lapisan embedding untuk kami:
Mungkin "good", "great", "fantastic" and "awesome"  ditempatkan berdekatan dan kata-kata seperti "bad", "horrible", 
"sucks" ditempatkan berdekatan. Kami juga berharap bahwa pengelompokan kata-kata ini
ditempatkan berjauhan satu sama lain yang menunjukkan bahwa mereka memiliki arti yang sangat berbeda.

----- GlobalPooling1D -----
Lapisan ini tidak istimewa dan hanya mengecilkan dimensi data kita untuk mempermudah komputasi model kita 
di lapisan selanjutnya. Karena lapisan embedding kami memetakan ribuan kata ke lokasi dalam ruang vektor,
mereka biasanya melakukan ini dalam ruang vektor berdimensi tinggi. Ini berarti ketika kita mendapatkan vektor 
kata dari lapisan embedding, mereka memiliki banyak dimensi dan dapat diperkecil oleh 
lapisan ini (GlobalPooling1D).

----- Dense -----
Dua lapisan terakhir di jaringan kami adalah lapisan Dense/padat yang terhubung penuh. Lapisan keluaran adalah 
salah satu neuron yang menggunakan fungsi sigmoid untuk mendapatkan nilai antara 0 dan 1 yang akan mewakili 
kemungkinan ulasan menjadi positif atau negatif. Lapisan sebelumnya berisi 16 neuron dengan fungsi aktivasi relu 
yang dirancang untuk menemukan pola antara kata-kata berbeda yang ada dalam ulasan.
"""