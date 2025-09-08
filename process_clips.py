# pip install tensorflow pandas numpy scikit-learn tqdm

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score

import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, 
                                     LSTM, TimeDistributed, BatchNormalization, concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# --- KONFIGURASI (Digunakan untuk arsitektur model) ---
# Pastikan parameter ini SAMA dengan yang digunakan saat pra-pemrosesan
CLIP_DURATION_S = 4
FRAMES_PER_CLIP = 16
IMG_SIZE = 64
SR = 22050
N_MELS = 128

# --- LANGKAH 1: Muat Metadata Klip yang Sudah Diproses ---
print("Memuat metadata klip yang sudah diproses...")
try:
    metadata_df = pd.read_csv("clip_metadata.csv")
except FileNotFoundError:
    print("\nERROR: File 'clip_metadata.csv' tidak ditemukan.")
    print("Pastikan Anda sudah menjalankan skrip 'preprocess_clips.py' terlebih dahulu.")
    exit()

# Encode label
le = LabelEncoder()
metadata_df['label_encoded'] = le.fit_transform(metadata_df['emotion'])
NUM_CLASSES = len(le.classes_)
print(f"Ditemukan {len(metadata_df)} klip data untuk {NUM_CLASSES} kelas emosi.")
print("Distribusi kelas:")
print(metadata_df['emotion'].value_counts())

# Pisahkan data menjadi train dan validation
train_df, val_df = train_test_split(
    metadata_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=metadata_df['label_encoded']
)

# --- LANGKAH 2: BUAT DATA GENERATOR YANG JAUH LEBIH CEPAT ---
class PreprocessedDataGenerator(Sequence):
    """Generator yang hanya memuat file .npz yang sudah diproses."""
    def __init__(self, dataframe, batch_size=32, shuffle=True):
        self.df = dataframe.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = self.df.index.tolist()
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        # Ambil indeks untuk batch saat ini
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Ambil baris dataframe untuk batch ini
        batch_df = self.df.iloc[batch_indices]
        
        # Hasilkan data
        X_video, X_audio, y = self._data_generation(batch_df)
        return (X_video, X_audio), y

    def on_epoch_end(self):
        # Acak indeks di setiap akhir epoch
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _data_generation(self, batch_df):
        # Dapatkan shape dari file pertama untuk inisialisasi
        # Ini aman karena semua klip memiliki dimensi yang sama
        first_file_path = batch_df.iloc[0]['filepath']
        with np.load(first_file_path) as data:
            video_shape = data['video'].shape
            audio_shape = data['audio'].shape

        # Buat array kosong untuk menampung batch data
        X_video = np.empty((self.batch_size, *video_shape), dtype=np.float32)
        X_audio = np.empty((self.batch_size, *audio_shape), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)
        
        # Isi array dengan data dari file .npz
        for i, (_, row) in enumerate(batch_df.iterrows()):
            try:
                with np.load(row['filepath']) as data:
                    # Muat dan normalisasi video (dari uint8 ke float32)
                    X_video[i,] = data['video'].astype(np.float32) / 255.0
                    X_audio[i,] = data['audio']
                    y[i] = row['label_encoded']
            except Exception as e:
                print(f"\nWarning: Gagal memuat file {row['filepath']}. Mengisi dengan nol. Error: {e}")
                X_video[i,] = np.zeros(video_shape, dtype=np.float32)
                X_audio[i,] = np.zeros(audio_shape, dtype=np.float32)
                y[i] = 0 # Label default jika error

        return X_video, X_audio, to_categorical(y, num_classes=NUM_CLASSES)

# --- LANGKAH 3: BANGUN ARSITEKTUR MODEL ---
def build_model(num_classes):
    """Membangun model dual-stream dari nol."""
    audio_time_steps = 1 + int(SR * CLIP_DURATION_S / 512)

    # Cabang Visual (Video)
    video_input = Input(shape=(FRAMES_PER_CLIP, IMG_SIZE, IMG_SIZE, 3), name='video_input')
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))(video_input)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Flatten())(x)
    video_features = LSTM(32)(x)
    
    # Cabang Audio
    audio_input = Input(shape=(N_MELS, audio_time_steps, 1), name='audio_input')
    y = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(audio_input)
    y = BatchNormalization()(y)
    y = MaxPooling2D((2, 2))(y)
    y = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D((2, 2))(y)
    audio_features = Flatten()(y)

    # Gabungkan (Fusion)
    merged = concatenate([video_features, audio_features])
    merged = Dropout(0.5)(merged)
    
    # Classifier
    z = Dense(64, activation='relu')(merged)
    z = Dropout(0.5)(z)
    output = Dense(num_classes, activation='softmax')(z)
    
    model = Model(inputs=[video_input, audio_input], outputs=output)
    return model

# --- BLOK EKSEKUSI UTAMA ---
if __name__ == '__main__':
    # Konfigurasi Training
    BATCH_SIZE = 32  # Bisa dinaikkan karena generator lebih ringan (coba 16, 32, atau 64)
    EPOCHS = 50

    # Inisialisasi generator
    train_generator = PreprocessedDataGenerator(train_df, batch_size=BATCH_SIZE, shuffle=True)
    val_generator = PreprocessedDataGenerator(val_df, batch_size=BATCH_SIZE, shuffle=False)
    
    # Definisikan struktur output untuk tf.data.Dataset
    audio_time_steps = 1 + int(SR * CLIP_DURATION_S / 512)
    output_signature = (
        (tf.TensorSpec(shape=(None, FRAMES_PER_CLIP, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(None, N_MELS, audio_time_steps, 1), dtype=tf.float32)),
        tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
    )

    # Buat tf.data.Dataset dari generator
    train_dataset = tf.data.Dataset.from_generator(lambda: train_generator, output_signature=output_signature)
    val_dataset = tf.data.Dataset.from_generator(lambda: val_generator, output_signature=output_signature)

    # Terapkan optimasi prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    # Bangun dan kompilasi model
    model = build_model(NUM_CLASSES)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Siapkan callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # --- LANGKAH 4: TRAINING ---
    print("\n--- Memulai Training ---")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping]
    )

    # --- LANGKAH 5: EVALUASI ---
    print("\n--- Training Selesai. Memulai Evaluasi ---")
    
    # Melakukan prediksi pada seluruh data validasi
    y_pred_probs = model.predict(val_dataset)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # Mengambil label asli dari dataframe validasi
    # Perlu berhati-hati karena batch terakhir mungkin tidak penuh
    num_val_samples = len(val_df)
    total_batches = len(val_generator)
    y_true_full = val_generator.df['label_encoded'].values
    y_true_truncated = y_true_full[:total_batches * BATCH_SIZE]
    
    # Pastikan jumlah prediksi dan label asli sama
    y_true = y_true_truncated[:len(y_pred_classes)]

    macro_f1 = f1_score(y_true, y_pred_classes, average='macro')
    print(f"\nMacro F1-Score (Validation): {macro_f1:.4f}")
    print("\nClassification Report (Validation):")
    print(classification_report(y_true, y_pred_classes, target_names=le.classes_, zero_division=0))