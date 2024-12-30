import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import tempfile
import os

class SpeechProcessor:
    def __init__(self):
        # Load Wav2Vec model and processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "facebook/wav2vec2-base-960h"  # English model
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
        
        # Supported sample rate by Wav2Vec2
        self.target_sample_rate = 16000
        
    def load_audio(self, audio_file):
        """Load and preprocess audio file"""
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_file:
            audio_file.save(temp_file.name)
            
            try:
                # Load the audio file
                waveform, sample_rate = torchaudio.load(temp_file.name)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample if necessary
                if sample_rate != self.target_sample_rate:
                    resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                    waveform = resampler(waveform)
                
                return waveform.squeeze().numpy()
            
            finally:
                os.unlink(temp_file.name)
    
    def transcribe_audio(self, audio_file):
        """Transcribe audio file to text using Wav2Vec"""
        try:
            # Load and preprocess audio
            audio_input = self.load_audio(audio_file)
            
            # Tokenize
            input_values = self.processor(
                audio_input, 
                sampling_rate=self.target_sample_rate, 
                return_tensors="pt"
            ).input_values.to(self.device)
            
            # Perform inference
            with torch.no_grad():
                logits = self.model(input_values).logits
            
            # Decode the predicted tokens to text
            predicted_ids = torch.argmax(logits, dim=-1)
            transcribed_text = self.processor.batch_decode(predicted_ids)[0]
            
            return {
                'text': transcribed_text,
                'success': True
            }
            
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            return {
                'text': None,
                'success': False,
                'error': str(e)
            }