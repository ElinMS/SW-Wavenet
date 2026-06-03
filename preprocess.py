import os
import glob
import librosa
import numpy as np
import torch
import multiprocessing

def process_file(args):
    audio_path, out_dir, sample_rate, n_fft, hop_length, n_mels = args
    basename = os.path.basename(audio_path)
    out_path = os.path.join(out_dir, basename.replace('.wav', '.pt'))
    
    if os.path.exists(out_path):
        return
        
    try:
        y, _ = librosa.load(audio_path, sr=sample_rate)
        
        # 1. Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sample_rate, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        spectrogram = torch.tensor(log_mel_spec, dtype=torch.float32).transpose(0, 1)
        
        # 2. Raw waveform
        waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        
        # Save to disk
        torch.save({'spectrogram': spectrogram, 'waveform': waveform}, out_path)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

def main():
    sample_rate = 16000
    n_fft = 1024
    hop_length = 512
    n_mels = 128
    
    dirs_to_process = [
        ("/home/teaching/Elin/fan/train", "/home/teaching/Elin/fan_preprocessed/train"),
        ("/home/teaching/Elin/fan/test", "/home/teaching/Elin/fan_preprocessed/test")
    ]
    
    tasks = []
    for in_dir, out_dir in dirs_to_process:
        os.makedirs(out_dir, exist_ok=True)
        wav_files = glob.glob(os.path.join(in_dir, "*.wav"))
        for wav_file in wav_files:
            tasks.append((wav_file, out_dir, sample_rate, n_fft, hop_length, n_mels))
            
    print(f"Found {len(tasks)} files to process.")
    
    # Use multiprocessing to speed up disk I/O and CPU computation
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for preprocessing...")
    
    pool = multiprocessing.Pool(processes=num_cores)
    for i, _ in enumerate(pool.imap_unordered(process_file, tasks)):
        if (i + 1) % 500 == 0 or (i + 1) == len(tasks):
            print(f"Processed {i+1}/{len(tasks)} files...")
    
    pool.close()
    pool.join()
    print("Done preprocessing!")

if __name__ == "__main__":
    main()
