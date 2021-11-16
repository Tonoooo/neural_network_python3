## disini kita hanya Memuat & Melihat Data, belum pake tensorflow
# Data sejauh ini merupakan bagian terpenting dari jaringan saraf mana pun. Memilih data yang tepat
# dan mengubahnya menjadi bentuk yang dapat digunakan dan dipahami oleh jaringan saraf sangat penting
# dan akan memengaruhi kinerja jaringan. Ini karena data yang kami lewati jaringan adalah apa yang akan digunakannya untuk mengubah bobot dan biasnya!

import tensorflow as tf
from tensorflow import keras  # keras adalah API untuk tensorflow. (didalam keras ada datasets fashion yang akan kitagunakan)
import numpy as np
import matplotlib.pyplot as plt

## di Keras ada datasets,Dataset yan gakan kita gunakan untuk memulai adalah dataset Fashion MNIST.
# Dataset ini berisi 60000 gambar item pakaian/pakaian yang berbeda
data = keras.datasets.fashion_mnist

## Sekarang kita akan membagi data kita menjadi data pelatihan dan pengujian.
# Hal ini penting untuk kita lakukan agar kita dapat menguji keakuratan model pada data yang belum pernah dilihat sebelumnya.
# ini sudah ada urutannya (ingat harus sesuai urutan, tapi nama variabelnya bebas):
# itu variabelnya sudah berurutan: veriabel pertama = melatih gambar, varibel kedua= melatih label, variabel ketiga = menguji gambar, variabel keempat= menguji label
(latih_gambar, latih_label), (uji_gambar, uji_label) = data.load_data()



# mendefinisikan daftar nama kelas
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# mengecilkan datanya agar membuat perhitungan lebih mudah untuk model, Kami melakukan ini dengan membagi setiap gambar, dengan 255.
latih_gambar = latih_gambar/255.0
uji_gambar = uji_gambar/255.0
print(latih_gambar[7])
# menampilkan gambarnya
plt.imshow(latih_gambar[7], cmap=plt.cm.binary) # [7]=index ke 7, cmap=plt.cm.binary = agar gambarnya terlihat lebih baik
plt.show()
# gambar 28x28 pixel, dan mereka termasuk kedalam array