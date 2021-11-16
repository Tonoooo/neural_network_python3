# Text Classification
# mengklasifikasikan ulasan film sebagai positif atau negatif.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# dataset, information = tfds.load('imdb_reviews/subwords8k', with_info=True,                  as_supervised=True)
# train_dataset, test_dataset = dataset['train'], dataset['test']
## Memuat data
imdb = keras.datasets.imdb # dataset film IMDB dari keras.

## membagi data nya
# untuk melatih         dan    untuk menguji
(latih_data, latih_label), (uji_data, uji_label) = imdb.load_data(num_words=10000)
    # karna kata didatasetsnya sangat banyak kita hanya menggunakan 10000 kata yang paling sering saja

## data yang Disandikan Bilangan Bulat (integer)
# Setelah melihat datanya, kami akan melihat bahwa ulasan nya dikodekan dengan integer. Ini berarti
# bahwa setiap kata dalam ulasan direpresentasikan sebagai integer positif di mana setiap integer
# mewakili kata tertentu. Ini diperlukan karena kita tidak dapat meneruskan string ke jaringan saraf kita.
# Namun, jika kita (sebagai manusia) ingin dapat membaca ulasan nya dan melihat seperti apa tampilannya,
# kita harus menemukan cara untuk mengubah ulasan yang disandikan bilangan bulat itu kembali menjadi string:
#print(latih_data[0]) # melihat datanya
kata_index = imdb.get_word_index() # ini akan memberi kita tuple yang memiliki string dan kata
# kita akan memecah tuple itu menjadi k dan v yang merupakan singkatan dari key dan value, key menjadi kata dan value menjadi integer
kata_index = {k:(v+3) for k, v in kata_index.items()}
    # jadi semua kata dalam datasets latih dan uji kita memiliki key dan value dimulai dari 1, jadi {k:(v+3) for k, v in kata_index.items()}
    # kita hanya menambahkan 3 ke semua valuenya (contohnya, key? memiliki value 1 lalu ditambah 3 jadi: key? valuenya 4).
# maka dariitu value 0:3 akan kosong, nah diisilah itu dengan ini:
kata_index["<PAD>"] = 0 #padding
kata_index["<START>"] = 1
kata_index["<UNK>"] = 2 #unknow / tidak tahu
kata_index["<UNUSED>"] = 3
    #padding untuk agar set film kita memiliki panjang yang sama.
# membalikan yang tadinya key value => jadi value key
dibalik_kata_index = dict([(value,key) for (key,value) in kata_index.items()])

## Data Prapemrosesan
# kita telah melihat panjang datanya berbeda. Ini adalah masalah. Kami tidak dapat mengirimkan data yg panjang
# berbeda ke jaringan saraf keluar. Oleh karena itu kita harus membuat setiap ulasan sama panjangnya. Untuk
# melakukan ini, kita akan mengikuti prosedur di bawah ini:
# - jika ulasan lebih dari 250 kata, potong kata tambahan
# - jika ulasan kurang dari 250 kata, tambahkan jumlah 's yang diperlukan agar sama dengan 250.
#   Untungnya bagi kami keras memiliki fungsi yang dapat melakukan ini untuk kami:
# kita akan mengunakan <PAD> tadi untuk menetapkan panjang semua data kita, terutama yang kurang dari 250 kata
# kita tambah kan <PAD> sampai nilai itu. JIka lebih 250 maka kita akan pangkas ulasan itu:
latih_data = keras.preprocessing.sequence.pad_sequences(latih_data, value=kata_index["<PAD>"], padding="post", maxlen=250)
uji_data= keras.preprocessing.sequence.pad_sequences(uji_data, value=kata_index["<PAD>"], padding="post", maxlen=250)
    # pad_sequences((masukan_datanya),(value/katanya),(dipost),(maximal_panjangnya_berapa))

## fungsi untuk memecahkan kode
# jadi latih dan uji menjadi kata yang dapat dibaca oleh manusia
def decode_review(text):
    return " ".join([dibalik_kata_index.get(i, "?") for i in text])
    # ini akan mengambalikan kepada kita semua keys yang kita inginkan
    # jadi kan dia akan mengeluarkan key yang berupa string, disetiap katanya akan ada spasi " ".
    # get(i, "?") => mendapatkan index i yang akan kita definisikan, jika tidak dapat menemukan tidak menemukan
    # nilai yang akan dilakukan adalah mengeluarkan "?"
""" metod .join() = mengambil semua item dalam iterable dan bergabung menjadi satu string.kalo tidak mengerti coba run ini:
    y = ("John", "Peter", "Vicky")
    x = "#".join(y)
    print(x) # setelah dirun => John#Peter#Vicky   # ditengahnya ada #"""
# kita coba membacanya
#print(len(uji_data[0]), len(uji_data[1])) # ini panjangnya 68 260, ini tidak akan berfungsi dimodel yang akan kita

## Mendefinisikan Modelnya
model = keras.Sequential()
model.add(keras.layers.Embedding(10000,16))
model.add(keras.layers.GlobalAveragePoolingID())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))