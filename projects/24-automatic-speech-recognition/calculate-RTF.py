import pandas as pd
import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T
import warnings
import time
from tqdm import tqdm
from utils import recognize_text, get_device, AverageMeter
from models import ASRModel


# Suppress warnings
warnings.filterwarnings("ignore")

# Load device and vocabulary
device = get_device()
vocab = torch.load('result/vocab.pt')

# Load model hyperparameters
with open('result/config/hyperparameters.txt', 'r') as f:
    params = eval(f.read())
params['vocab'] = vocab
params['vocab_size'] = len(vocab)

# Initialize and load the ASR model
model = ASRModel(**params, device=device).to(device)
model.load_state_dict(torch.load('result/model/best_model.pt', weights_only=True))

# Load audio files for testing
files = pd.read_csv('C:/Users/ahmad/Documents/1.AI/11.ASR Persian/3.Dataset/split/test-ann.csv')['audio'].values

# Initialize RTF (Real-Time Factor) meter
rtf = AverageMeter()

# Process each audio file
with tqdm(files, desc='Processing audios') as tepoch:
    for file in tepoch:
        start_time = time.time()

        # Load and preprocess the audio file
        waveform, sample_rate = torchaudio.load(f"C:/Users/ahmad/Documents/1.AI/11.ASR Persian/3.Dataset/audios/{file}.wav")
        spec = nn.Sequential(
            T.Resample(orig_freq=sample_rate, new_freq=16000),
            T.MelSpectrogram(n_mels=80, n_fft=400)
        ).requires_grad_(False)(waveform)
        spec = torch.log10(spec + 1e-9)  # Log scaling

        # Recognize text from the spectrogram
        text = recognize_text(model, spec, vocab, device)

        # Calculate and update RTF
        audio_duration = waveform.shape[1] / sample_rate
        processing_time = time.time() - start_time
        rtf.update(processing_time / audio_duration)

        # Update progress bar with current RTF
        tepoch.set_postfix(rtf=rtf.avg)

# Print the average RTF
print(f"Average Real-Time Factor (RTF): {rtf.avg}")