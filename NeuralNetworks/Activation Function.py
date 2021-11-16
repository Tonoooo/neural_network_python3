"""
-------- Disini kita hanya menjelaskan tentang Activation Function ----------

    Kenapa harus menggunakan Activation Function pada NN?
Sebenarnya digunakan untuk menentukan ouput dari si NN seperti yes or no. Fungsi yang memetakan
input menjadi 0 sampai 1 atau -1 sampai 1 (tergantung fungsinya).

    Pada dasarnya Activation Function dibedakan menjadi 2 tipe:
1. Linear Activation Function = yang garisnya tegak/lurus
2. Non-linear Activation Function = yang garisnya melengkung

        Macam macam fungsi aktifasi:
-Sigmoid Function terlihat seperti S-shape => sigmoid
    Alasan utama kenapa menggunakan function sigmoid adalah karena nilainya ada di
    range (0 sampai 1). Sehingga, bisa digunakan oleh model untuk memprediksi probability s
    ebagai output. Karena probability juga nilainya ada di range (0 sampai 1).
-Tanh atau hyperbolic tangent Activation Function => tanh
    Tanh juga seperti logistic sigmoid tapi lebih bagus. Range dari tanh function berasal
    dari (-1 sampai 1). Tanh juga berbentuk sigmoidal (s).
-ReLU (Rectified Linear Unit) Activation Function => relu
    ReLU merupakan function yang paling digunakan sampai sekarang, ini digunakan di hampir
    semua convex atau deep learning lainnya.
    Range: [ 0 sampai takhingga)
-Softmax => softmax
    fungsi softmax adalah ia akan memilih nilai untuk setiap neuron sehingga semua nilai
    itu bertambah menjadi satu

"""