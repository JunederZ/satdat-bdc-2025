import numpy as np
import pandas as pd
import os
import cv2
import librosa
from tqdm import tqdm

# --- KONFIGURASI (Sama seperti skrip training) ---
DATA_DIR = "videos"
CSV_PATH = "datatrain.csv"
OUTPUT_CLIPS_DIR = "processed_clips" # Folder baru untuk menyimpan hasil

CLIP_DURATION_S = 4
FRAMES_PER_CLIP = 16
IMG_SIZE = 64
SR = 22050
N_MELS = 128

def preprocess_and_save_clips():
    # Buat folder output jika belum ada
    os.makedirs(OUTPUT_CLIPS_DIR, exist_ok=True)
    
    # 1. Bersihkan dan siapkan dataframe
    print("Membaca dan membersihkan dataframe...")
    label_map = {
        'surprise': 'Surprise', 'terkejut': 'Surprise', 'trkejut': 'Surprise', 'kaget': 'Surprise', 'terkjut': 'Surprise',
        'proud': 'Proud', 'bangga': 'Proud',
        'trust': 'Trust', 'percaya': 'Trust', 'faith': 'Trust',
        'sadness': 'Sadness', 'sad': 'Sadness',
        'anger': 'Anger', 'marah': 'Anger',
        'joy': 'Joy',
        'fear': 'Fear',
        'neutral': 'Neutral'
    }
    VALID_EMOTIONS = {'Proud', 'Trust', 'Joy', 'Surprise', 'Neutral', 'Sadness', 'Fear', 'Anger'}
    df = pd.read_csv(CSV_PATH)
    def clean_emotion(emotion):
        if not isinstance(emotion, str): return None
        clean_key = emotion.strip().lower()
        return label_map.get(clean_key)
    df['emotion_clean'] = df['emotion'].apply(clean_emotion)
    df_cleaned = df.dropna(subset=['emotion_clean'])
    df_cleaned = df_cleaned[df_cleaned['emotion_clean'].isin(VALID_EMOTIONS)]
    df_cleaned['video_path'] = df_cleaned['id'].apply(lambda x: os.path.join(DATA_DIR, f"{x}.mp4"))
    df_cleaned = df_cleaned[df_cleaned['video_path'].apply(os.path.exists)]

    # 2. Loop melalui setiap video dan buat klip
    clip_metadata = []
    audio_time_steps = 1 + int(SR * CLIP_DURATION_S / 512)

    for index, row in tqdm(df_cleaned.iterrows(), total=len(df_cleaned), desc="Processing Videos"):
        video_path = row['video_path']
        video_id = row['id']
        emotion_label = row['emotion_clean']
        
        try:
            # Dapatkan info durasi
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if fps is None or total_frames is None or fps <= 0 or total_frames <= 0:
                continue

            duration = total_frames / fps
            num_clips = int(duration // CLIP_DURATION_S)

            # Muat seluruh audio sekali saja untuk efisiensi
            full_audio, _ = librosa.load(video_path, sr=SR)

            for i in range(num_clips):
                start_time = i * CLIP_DURATION_S
                
                # --- Ekstrak Audio untuk klip ini ---
                start_sample = int(start_time * SR)
                end_sample = start_sample + int(CLIP_DURATION_S * SR)
                audio_clip = full_audio[start_sample:end_sample]
                
                if len(audio_clip) < SR * CLIP_DURATION_S:
                    audio_clip = np.pad(audio_clip, (0, SR * CLIP_DURATION_S - len(audio_clip)))
                
                melspec = librosa.feature.melspectrogram(y=audio_clip, sr=SR, n_mels=N_MELS)
                log_melspec = librosa.power_to_db(melspec, ref=np.max)
                
                if log_melspec.shape[1] != audio_time_steps:
                    # Lakukan padding/truncate jika perlu (jarang terjadi tapi penting)
                    if log_melspec.shape[1] < audio_time_steps:
                        log_melspec = np.pad(log_melspec, ((0,0), (0, audio_time_steps - log_melspec.shape[1])))
                    else:
                        log_melspec = log_melspec[:, :audio_time_steps]

                audio_features = np.expand_dims(log_melspec, axis=-1)

                # --- Ekstrak Video Frame untuk klip ini ---
                video_frames = np.zeros((FRAMES_PER_CLIP, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) # Gunakan uint8 untuk hemat space
                cap = cv2.VideoCapture(video_path)
                start_frame = int(start_time * fps)
                frame_indices = np.linspace(start_frame, start_frame + (CLIP_DURATION_S * fps) - 1, FRAMES_PER_CLIP, dtype=int)
                
                for j, frame_idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                        video_frames[j] = frame
                cap.release()
                
                # --- Simpan ke file .npz ---
                clip_filename = f"{video_id}_clip_{i}.npz"
                clip_filepath = os.path.join(OUTPUT_CLIPS_DIR, clip_filename)
                
                np.savez_compressed(clip_filepath, video=video_frames, audio=audio_features)
                
                clip_metadata.append({'filepath': clip_filepath, 'emotion': emotion_label})

        except Exception as e:
            print(f"\nError processing video {video_path}: {e}")

    # 3. Simpan metadata ke CSV untuk digunakan saat training
    metadata_df = pd.DataFrame(clip_metadata)
    metadata_df.to_csv("clip_metadata.csv", index=False)
    print(f"\nPra-pemrosesan selesai. {len(metadata_df)} klip disimpan di folder '{OUTPUT_CLIPS_DIR}'.")
    print(f"File metadata 'clip_metadata.csv' telah dibuat.")

if __name__ == '__main__':
    preprocess_and_save_clips()