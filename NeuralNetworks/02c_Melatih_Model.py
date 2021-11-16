# Text Classification

from tensorflow import keras
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
model = keras.Sequential()
model.add(keras.layers.Embedding(10000,16)) # 10000 neuron dengan 16 dimensi
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu")) # l6 neuron
model.add(keras.layers.Dense(1, activation="sigmoid")) # 1 output neuron
model.summary()

### ------- Mengcompile dan Melatih Model --------
## Mengcompile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # loss="binary_crossentropy" = kan kita memiliki neuron yang mengeluarkan 0 /1 jadi yang terjadi adalah
    # menghitung kerugian perbedaan jarak. Misalkan output 0.2 menjadi 0, nah ini dihitung loss nya

## Data Validasi
# Untuk model khusus ini, kami akan memperkenalkan ide baru tentang data validasi. Dalam tutorial terakhir
# ketika kami melatih akurasi model setelah setiap epoch pada data pelatihan saat ini, data yang telah dilihat
# model sebelumnya. Ini bisa menjadi masalah karena sangat mungkin model hanya dapat menghafal data input dan
# output terkait dan akurasi akan mempengaruhi bagaimana model dimodifikasi saat train. Jadi untuk menghindari
# masalah ini kami akan membagi data pelatihan kami menjadi dua bagian, pelatihan dan validasi. Model akan
# menggunakan data validasi untuk memeriksa akurasi setelah belajar dari data pelatihan. Ini diharapkan akan
# menghasilkan kita menghindari kepercayaan palsu untuk model kita.
# Kami dapat membagi data pelatihan kami menjadi data validasi seperti:
x_val = latih_data[:10000] # ada 25000 data diulasannya, kita hanya menggunakan 10000 saja
x_latih = latih_data[10000:]
y_val = latih_label[:10000]
y_latih = latih_label[10000:]

## Melatih Model
model.fit(x_latih, y_latih, epochs=40, batch_size=512, validation_data=(x_val,y_val), verbose=1)
    # batch_size=512 = berpa banyak ulasan yang akan kita muat sekaligus

## Menguji Model
# Untuk melihat hasil akurasi
hasil = model.evaluate(uji_data,uji_label)
print(hasil)

## Menerapkan
uji_ulasan = uji_data[0]
predik = model.predict([uji_ulasan])
print(">>>>>>>> ulasan: ")
print(decode_review(uji_ulasan))
prediksi = int(predik[0]) # [0] = index pertama
penguji = int(uji_label[0])
def kebenaran(ulasan, siapa):
    if (ulasan == 0):
        print("Hasil ", siapa," : negatif ")
    elif (ulasan == 1):
        print("Hasil ", siapa," : positif")
    else:
        print("Hasil ", siapa," : bodo")

kebenaran(prediksi,"prediksi")
kebenaran(penguji,'penguji')