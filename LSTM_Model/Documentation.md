# Penjelasan LSTM-Autoencoder untuk Deteksi Anomali

Kode yang telah dibuat mengimplementasikan model LSTM-Autoencoder untuk deteksi anomali pada data uptime. Model ini disimpan di direktori `uptime_lstm/` dan dapat digunakan untuk mendeteksi pola tidak normal (anomali) pada data uptime server. Berikut penjelasan detail tentang model dan komponen-komponennya:

## 1. Apa itu Autoencoder?

Autoencoder adalah jenis neural network yang dilatih untuk merekonstruksi input aslinya. Autoencoder terdiri dari dua bagian utama:

- **Encoder**: Memetakan input ke representasi laten (compressed representation).
- **Decoder**: Memetakan representasi laten kembali ke dimensi input asli.

Prinsip kerja autoencoder untuk deteksi anomali adalah sebagai berikut:
1. Model dilatih untuk merekonstruksi data normal.
2. Ketika diberi data anomali, model akan kesulitan merekonstruksi data tersebut.
3. Reconstruction error yang tinggi mengindikasikan anomali.

## 2. Komponen LSTM (Long Short-Term Memory)

LSTM adalah jenis arsitektur Recurrent Neural Network (RNN) yang dirancang untuk mengatasi masalah vanishing gradient yang sering terjadi pada RNN biasa. LSTM cocok untuk data sequential dan time series karena:

1. **Memory Cell**: Menyimpan informasi untuk periode waktu yang lama
2. **Gates**: 
   - **Forget Gate**: Memutuskan informasi mana yang dibuang dari memory cell
   - **Input Gate**: Memutuskan nilai baru mana yang disimpan dalam memory cell
   - **Output Gate**: Mengontrol bagian mana dari memory cell yang dioutputkan

## 3. Penggabungan Database dengan LSTM-Autoencoder

Kode ini sekarang telah diintegrasikan dengan sistem basis data untuk mengambil data uptime langsung dari database. Berikut adalah komponen-komponen yang ditambahkan:

1. **Database Connection**: Menggunakan modul `pyodbc` untuk menghubungkan ke database SQL Server
2. **Konfigurasi**: Membaca konfigurasi koneksi dari file `HashedDBInfo.json`
3. **Query Data**: Mengambil data uptime dari tabel `tbl_t_firewall_uptime`
4. **Fallback Mechanism**: Jika koneksi ke database gagal, kode akan menggunakan data dummy untuk testing

## 4. Arsitektur LSTM-Autoencoder pada Kode

### Encoder
```python
encoder_inputs = Input(shape=(sequence_length, input_dim))
encoder = LSTM(64, activation='relu', return_sequences=True)(encoder_inputs)
encoder = Dropout(0.2)(encoder)
encoder = LSTM(32, activation='relu', return_sequences=False)(encoder)
encoder = Dropout(0.2)(encoder)
```

Bagian encoder terdiri dari:
- Layer input dengan bentuk (sequence_length, input_dim)
- Layer LSTM pertama dengan 64 unit dan aktivasi ReLU
- Layer Dropout 20% untuk mencegah overfitting
- Layer LSTM kedua dengan 32 unit, tanpa return_sequences
- Layer Dropout 20% lagi untuk mencegah overfitting

### Representasi Laten (Bottleneck)
```python
latent_dim = 16
latent_representation = Dense(latent_dim)(encoder)
```

Representasi laten (compressed) memiliki dimensi 16, lebih kecil dari dimensi input untuk memaksa model belajar representasi yang efisien dari data.

### Decoder
```python
decoder = RepeatVector(sequence_length)(latent_representation)
decoder = LSTM(32, activation='relu', return_sequences=True)(decoder)
decoder = Dropout(0.2)(decoder)
decoder = LSTM(64, activation='relu', return_sequences=True)(decoder)
decoder = Dropout(0.2)(decoder)
decoder_outputs = TimeDistributed(Dense(input_dim))(decoder)
```

Bagian decoder terdiri dari:
- Layer RepeatVector yang mengembalikan dimensi sequence
- Layer LSTM dengan 32 unit, aktivasi ReLU
- Layer Dropout 20%
- Layer LSTM dengan 64 unit, aktivasi ReLU
- Layer Dropout 20%
- Layer TimeDistributed Dense untuk rekonstruksi output ke dimensi input asli

## 5. Proses Deteksi Anomali

1. **Preprocessing**: Data dinormalisasi dengan MinMaxScaler dan dibagi menjadi sequence
2. **Training**: Model dilatih untuk meminimalkan error rekonstruksi pada data normal
3. **Threshold**: Nilai threshold (ambang batas) ditentukan berdasarkan persentil dari reconstruction error
4. **Deteksi**: Data dengan reconstruction error melebihi threshold diidentifikasi sebagai anomali
5. **Visualisasi**: Hasil deteksi anomali divisualisasikan dalam berbagai grafik