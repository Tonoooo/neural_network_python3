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

## Membuat Model
# -Sekarang saatnya untuk membuat model jaringan saraf pertama kami! Kami akan melakukan ini dengan menggunakan
# -objek Sequential dari keras. Model Sequential hanya mendefinisikan urutan lapisan dimulai dengan lapisan input
# -dan diakhiri dengan lapisan output. Model kami akan memiliki 3 lapisan, dan lapisan input dari 784 neuron
# -(mewakili semua 28x28 piksel dalam gambar) lapisan tersembunyi dari 128 neuron yang berubah-ubah dan lapisan
# -output dari 10 neuron yang mewakili kemungkinan gambar menjadi masing-masing dari 10 kelas.
#Hal pertama yang kita lakukan adalha mendefinisikan arsitektur/lapisan untuk model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # ini untuk input layer, dengan neuron 784. flatter untuk meratakan
    keras.layers.Dense(128,activation="relu"), #ini untuk hidden layers, dengan neuron 128, layers denses/layers padat
    keras.layers.Dense(10,activation="softmax") #ini untuk output layers, dengan neuron 10, mengunakan fungsi aktivasi softmax
    ])
    #-Sequential = sekuensial memungkinkan Anda membuat model lapis demi lapis untuk sebagian besar masalah. Ini terbatas karena
    #   tidak memungkinkan Anda membuat model yang berbagi lapisan atau memiliki banyak input atau output.
    #   jadi Sequential tidak bisa memiliki banyak layer output.
    #-Flatter = jadi misal saat menyampaikan datanya berbentuk 2d/3d array, maka gunakan Flatter agar datanya dapat dilewati/dicerna oleh setiap neuron:
    #   kan datanya begini [[0],[0.25],[0..],..] mengunakan Flatter agar diratakan jadi [0, 0.25, 0,...]
    #   (kurung siku [] yang ada didalamnya hilang).
    #-layers.Deses = layers padat pada dasarnya berarti terhubung sepenuhnya dengan neuron lain,
    #   maksudnya setiap neuron terhubung ke neuron lain dijaringan berikutnya
    #-activation(funsi aktivasi)=  artinya mau mengunakan fungsi activasi apa, ada relu,softmax,sigmoid,dan lain...
    #-"relu" adalah fungsi yang digunakan difungsi aktivasi
    #   relu itu termasuk linear, dan relu merupakan fungsi aktivasi yang sangat cepat dan berfungsi sangat baik, range 0 sampai tek terhingga
    #-"softmax" ia juga merupakan fungsi akrivasi
    #   fungsi softmax adalah ia akan memilih nilai untuk setiap neuron sehingga semua nilai itu bertambah menjadi satu

## mengcompile modelnya
# Mengkompilasi model hanya memilih beberapa parameter untuk model kita seperti pengoptimal, fungsi kerugian, dan metrik untuk dilacak
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # jadi itu untuk melihat akurasi dari model kita dan melihat yang loss nya.
    # optimizer="adam" adalah algoritma optimasi yang dapat digunakan sebagai pengganti
    # prosedur penurunan gradien stokastik klasik untuk memperbarui bobot jaringan secara
    # iteratif berdasarkan data pelatihan.

## melatih model
model.fit(latih_gambar,latih_label,epochs=5)
    #-epochs adalah berapa kali model akan dilatih,
    #   cara kerja epochs adalah ia akan memilih gambar dan labels SECARA ACAK dan itu akan menyalurkannya melalui neuralnetwork
    #   jadi epochs yang dilakukannya: mengulangi latihan model dengan memilih gambar dan labels SECARA ACAK sampai beberapa kali ulangan
    #   epochs=5   ini akan mengulangi latihannya sampai 5 kali
    # PENTING!!!:
    #   Menggunakan banyak epochs(latihannya diulang) tidak akan menjadikan model menjadikan sangat sangat baik/sempurna
    #   karna epochs tidak melatih model dengan seluruh gambar dan label, ia hanya MEMILIH SECARA ACAK gambar dan labelnya.
    #   iya sih.. menggunakan banyak epoch menjadi labih baik tapi hanya sedikit.
    #   epochs=5 ini bukan berarti modelkita menjadi 5 kali lebih baik

## Menguji Model
# setelah model dilatih, sekarang saatnya untuk menguji keakuratannya.
# ini akan membuat 2 variabel, dan ini sudah ada urutannya:
# variabel pertama=untuk test loss, variabel kedua=untuk test akurasi
test_loss, test_acc = model.evaluate(uji_gambar, uji_label) #evaluate untuk menguji
print('\nMenguji akurasinya: ', test_acc)
    # setelah diuji/dirun: mengapa hasil akurasi tidak sama/lebih buruk dengan akurasi epoch terakhir?
    #   karna epoch tidak melatih model dengan seluruh gambar dan label, ia hanya MEMILIH SECARA ACAK gambar dan labelnya