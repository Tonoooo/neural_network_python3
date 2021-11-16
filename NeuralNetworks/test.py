# file test ini hanya sebagai tempat digunakan untuk mengetest / menguji sesuatu

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
model.save("modelnya.h5") # beri nama apa pun yang Anda inginkan tetapi diakhiri dengan .h5
"""

model = keras.models.load_model("model.h5")
def review_encode(s):
	encoded = [1]
	for word in s:
		if word.lower() in kata_index:
			encoded.append(kata_index[word.lower()])
		else:
			encoded.append(2)
	return encoded
with open("test.txt", encoding="utf-8") as f:
	for line in f.readlines():
		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
		encode = review_encode(nline)
		encode = keras.preprocessing.sequence.pad_sequences([encode], value=kata_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
		predict = model.predict(encode)
		print(line)
		print(encode)
		print(predict[0])