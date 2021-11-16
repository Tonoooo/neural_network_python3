# Saving & Loading Models
from tensorflow import keras

data = keras.datasets.imdb
(latih_data, latih_label), (uji_data, uji_label) = data.load_data(num_words=88000) # diubah menjadi 88000 karna kita akan mempreedik ulsan lain
kata_index = data.get_word_index()
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
"""
model = keras.Sequential()
model.add(keras.layers.Embedding(88000,16))  # diubah menjadi 88000 karna kita akan mempreedik ulsan lain
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu")) # l6 neuron
model.add(keras.layers.Dense(1, activation="sigmoid")) # 1 output neuron
model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
x_val = latih_data[:10000]
x_latih = latih_data[10000:]
y_val = latih_label[:10000]
y_latih = latih_label[10000:]
model.fit(x_latih, y_latih, epochs=40, batch_size=512, validation_data=(x_val,y_val), verbose=1)
hasil = model.evaluate(uji_data,uji_label)
print(hasil)

## ------- Menyimpan Model -------
model.save("model.h5") # beri nama apa pun yang Anda inginkan tetapi diakhiri dengan .h5
    # h5 seperti ektensi dikeras untuk save model
"""

## Memuat model
#Sekarang kita telah menyimpan model terlatih, kita tidak perlu melatihnya kembali! Kita cukup memuat model
# yang disimpan dengan menggunakan yang berikut ini. Cukup pastikan bahwa file .h5 berada di direktori yang
# sama dengan skrip python Anda.
model = keras.models.load_model("model.h5")

## Membuat Prediksi
# Sekarang saatnya menggunakan model tersimpan untuk membuat prediksi. Sekarang ini sedikit lebih sulit
# daripada yang terlihat karena kita perlu mempertimbangkan hal berikut:
# – model kita menerima data yang dikodekan bilangan bulat
# – model kita membutuhkan tinjauan yang panjangnya 250 kata
# Ini berarti kita tidak bisa begitu saja melewatkan string teks apa pun ke dalam model kita.
# Itu perlu dibentuk kembali dan direformasi untuk memenuhi kriteria di atas.

# Untuk memulai kita perlu mengkodekan data integer. Kami akan melakukan ini menggunakan fungsi berikut:
def review_encode(s):
    encoded = [1] # [1] karna semua data dimulai dari 1 maka kita akan mulai dengan 1, 1 adalah ini (kata_index["<START>"] = 1)
    # kita akan mengulang setiap kata yand ada di s (nline) dan akan mencari nomor yg terkait dengan kata kata itu dan akan menambahkanya kedalam list encoded
    for word in s:
        if word.lower() in kata_index:
            encoded.append(kata_index[word.lower()])
        else:
            encoded.append(2)
    return encoded
    # jadi didalam if kita akan memeriksa di kata tersebut ada di kota_index(semua kata yg sesuai dengan semua angka
    # yang mewakili kata kata itu). jika buakan kita akan menambahkan 2(kata_index["<UNK>"] = 2), sehingga program
    # tau bahwa itu kata yang tidak dikenal.
    # lower()method mengembalikan string di mana semua karakter huruf kecil.jadi hurufnya pada kecil gak ada yang besat

# Selanjutnya kita akan membuka file teks kita, membaca setiap review (dalam hal ini hanya satu) dan
# menggunakan model untuk memprediksi apakah itu positif atau negatif.
# kita mengunakan with karna agar tidak usah menutup file lagi sesudah dibuka
with open("test.txt", encoding="utf-8") as f: #encoding="utf-8" ini hanya code tesk standar
    for line in f.readlines():
        #kita akan mengganti,menghapus,membadi
        #  replace("str_lama","str_baru",count)Metode menggantikan frase tertentu dengan kalimat/sesuatu lain yang ditentukan.
        #  split()Metode membagi string ke dalam list.
        #  strip()Metode menghapus setiap terkemuka (spasi di awal) dan membuntuti (spasi di akhir) karakter (ruang default karakter untuk menghapus terkemuka)
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
        # kita akan mengkodekan (string jadi int)
        encode = review_encode(nline) # ini menggunakan fungsi yang diatas
        #  memangkas data menjadi 250 kata
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=kata_index["<PAD>"], padding="post",maxlen=250)        # menggunakan model untuk memprediksi
        predict = model.predict(encode)
        print(line) # print data asli
        print(encode) # print hasil pengkodean
        print(predict[0]) # print hasil prediksi modelnya
        # setelah dirun: kemungkinan hasil predik sekitar [0.9762057] atau [0.96..].
        # 0 = negatif, 1 = positif. misal hasilnya 0.96 berarti karna 0.96 mendekati 1 maka ulasannya positif
