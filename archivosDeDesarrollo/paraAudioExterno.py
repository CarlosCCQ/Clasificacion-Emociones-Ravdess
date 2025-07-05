import os
import time
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import librosa.display
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

class ExternalAudioTester:
    def __init__(self, model_path):
        self.model_path = model_path
        self.models = {}
        self.emotion_labels = {
            0: 'Neutral', 1: 'Calm', 2: 'Happy', 3: 'Sad',
            4: 'Angry', 5: 'Fearful', 6: 'Disgust', 7: 'Surprised'
        }
        self.emotion_colors = {
            'Neutral': '#808080', 'Calm': '#87CEEB', 'Happy': '#FFD700', 'Sad': '#4169E1',
            'Angry': '#FF4500', 'Fearful': '#800080', 'Disgust': '#228B22', 'Surprised': '#FF69B4'
        }
        
    def load_models(self):
        dt_path = os.path.join(self.model_path, "decision_tree_40mfccs.joblib")
        rf_path = os.path.join(self.model_path, "random_forest_40mfccs.joblib")
        cnn_path = os.path.join(self.model_path, "Emotion_Voice_Detection_Model_CNN.h5")
        
        if os.path.exists(dt_path):
            self.models['Decision Tree'] = joblib.load(dt_path)
            print("Decision Tree loaded")
        
        if os.path.exists(rf_path):
            self.models['Random Forest'] = joblib.load(rf_path)
            print("Random Forest loaded")
        
        if os.path.exists(cnn_path):
            self.models['CNN'] = load_model(cnn_path)
            print("CNN loaded")
    
    def extract_features(self, file_path, target_sr=22050, duration=3, max_length=130):
        try:
            y, sr = librosa.load(file_path, sr=target_sr, duration=duration)
            
            if len(y) < target_sr * 0.5:
                print(f"Audio too short: {len(y)/sr:.2f}s")
                return None, None, None
            
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            delta = librosa.feature.delta(mfccs)
            combined = np.vstack([mfccs, delta]).T
            
            padded = pad_sequences([combined], maxlen=max_length, 
                                 dtype="float32", padding="post", truncating="post")
            
            return padded[0], padded[0].flatten(), y
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None, None, None
    
    def predict_emotion(self, file_path, show_waveform=True, show_probabilities=True):
        print(f"\nAnalyzing: {os.path.basename(file_path)}")
        print("="*60)
        
        feat_2d, feat_flat, audio_data = self.extract_features(file_path)
        
        if feat_2d is None:
            return None
        
        if show_waveform:
            self.plot_waveform(audio_data, file_path)
        
        predictions = {}
        
        for model_name, model in self.models.items():
            start_time = time.time()
            
            try:
                if model_name == 'CNN':
                    X_input = feat_2d.reshape(1, feat_2d.shape[0], feat_2d.shape[1], 1)
                    prediction = model.predict(X_input, verbose=0)
                    pred_class = np.argmax(prediction[0])
                    confidence = np.max(prediction[0])
                    probabilities = prediction[0]
                else:
                    X_input = feat_flat.reshape(1, -1)
                    pred_class = model.predict(X_input)[0]
                    probabilities = model.predict_proba(X_input)[0]
                    confidence = np.max(probabilities)
                    
                    if isinstance(pred_class, str):
                        pred_class = int(pred_class) - 1
                
                inference_time = time.time() - start_time
                pred_emotion = self.emotion_labels.get(pred_class, "Unknown")
                
                predictions[model_name] = {
                    'emotion': pred_emotion,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'inference_time': inference_time
                }
                
                print(f"{model_name:15} → {pred_emotion:10} ({confidence:.3f}) [{inference_time*1000:.1f}ms]")
                
            except Exception as e:
                print(f"{model_name:15} → Error: {str(e)}")
                predictions[model_name] = None
        
        if show_probabilities:
            self.plot_probabilities(predictions)
        
        return predictions
    
    def plot_waveform(self, audio_data, file_path, sr=22050):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        librosa.display.waveshow(audio_data, sr=sr)
        plt.title(f'Waveform: {os.path.basename(file_path)}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        plt.subplot(1, 2, 2)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.colorbar()
        plt.title('MFCC Features')
        plt.xlabel('Time (s)')
        plt.ylabel('MFCC Coefficients')
        
        plt.tight_layout()
        plt.show()
    
    def plot_probabilities(self, predictions):
        n_models = len([p for p in predictions.values() if p is not None])
        if n_models == 0:
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        model_idx = 0
        for model_name, pred_data in predictions.items():
            if pred_data is None:
                continue
                
            emotions = list(self.emotion_labels.values())
            probabilities = pred_data['probabilities']
            colors = [self.emotion_colors[emotion] for emotion in emotions]
            
            bars = axes[model_idx].bar(emotions, probabilities, color=colors, alpha=0.7)
            axes[model_idx].set_title(f'{model_name}\nPredicted: {pred_data["emotion"]}')
            axes[model_idx].set_ylabel('Probability')
            axes[model_idx].set_ylim(0, 1)
            axes[model_idx].tick_params(axis='x', rotation=45)
            
            max_prob_idx = np.argmax(probabilities)
            bars[max_prob_idx].set_alpha(1.0)
            bars[max_prob_idx].set_edgecolor('black')
            bars[max_prob_idx].set_linewidth(2)
            
            model_idx += 1
        
        plt.tight_layout()
        plt.show()
    
    def batch_predict(self, audio_folder, audio_extensions=['.wav', '.mp3', '.flac', '.m4a']):
        print(f"\nBatch processing folder: {audio_folder}")
        print("="*60)
        
        if not os.path.exists(audio_folder):
            print(f"Folder not found: {audio_folder}")
            return {}
        
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend([f for f in os.listdir(audio_folder) if f.lower().endswith(ext)])
        
        if not audio_files:
            print(f"No audio files found in {audio_folder}")
            return {}
        
        print(f"Found {len(audio_files)} audio files")
        
        batch_results = {}
        
        for i, filename in enumerate(audio_files, 1):
            file_path = os.path.join(audio_folder, filename)
            print(f"\n[{i}/{len(audio_files)}] Processing: {filename}")
            
            predictions = self.predict_emotion(file_path, show_waveform=False, show_probabilities=False)
            batch_results[filename] = predictions
        
        self.summarize_batch_results(batch_results)
        return batch_results
    
    def summarize_batch_results(self, batch_results):
        print(f"\nBATCH RESULTS SUMMARY")
        print("="*60)
        
        model_predictions = {}
        
        for filename, predictions in batch_results.items():
            if predictions is None:
                continue
                
            for model_name, pred_data in predictions.items():
                if pred_data is None:
                    continue
                    
                if model_name not in model_predictions:
                    model_predictions[model_name] = {}
                
                emotion = pred_data['emotion']
                if emotion not in model_predictions[model_name]:
                    model_predictions[model_name][emotion] = 0
                model_predictions[model_name][emotion] += 1
        
        for model_name, emotion_counts in model_predictions.items():
            print(f"\n{model_name}:")
            total = sum(emotion_counts.values())
            for emotion, count in sorted(emotion_counts.items()):
                percentage = (count / total) * 100
                print(f"   {emotion:10} → {count:2d} files ({percentage:.1f}%)")
    
    def compare_models_on_file(self, file_path):
        print(f"\nDETAILED COMPARISON")
        print("="*60)
        
        predictions = self.predict_emotion(file_path, show_waveform=True, show_probabilities=True)
        
        if predictions is None:
            return None
        
        print(f"\nCONFIDENCE COMPARISON:")
        confidences = [(name, pred['confidence']) for name, pred in predictions.items() if pred is not None]
        confidences.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model_name, confidence) in enumerate(confidences, 1):
            emotion = predictions[model_name]['emotion']
            print(f"   {i}. {model_name:15} → {emotion:10} ({confidence:.3f})")
        
        return predictions
    
    def real_time_predict(self, file_path, segment_duration=2.0):
        print(f"\nREAL-TIME STYLE PREDICTION")
        print("="*60)
        
        try:
            y, sr = librosa.load(file_path, sr=22050)
            total_duration = len(y) / sr
            
            if total_duration < segment_duration:
                print(f"Audio too short for segmentation: {total_duration:.2f}s")
                return self.predict_emotion(file_path)
            
            segment_samples = int(segment_duration * sr)
            n_segments = int(total_duration / segment_duration)
            
            print(f"Audio duration: {total_duration:.2f}s")
            print(f"Analyzing {n_segments} segments of {segment_duration}s each")
            
            segment_results = {}
            
            for i in range(n_segments):
                start_idx = i * segment_samples
                end_idx = start_idx + segment_samples
                segment = y[start_idx:end_idx]
                
                temp_file = f"temp_segment_{i}.wav"
                sf.write(temp_file, segment, sr)
                
                print(f"\nSegment {i+1}/{n_segments} ({i*segment_duration:.1f}s - {(i+1)*segment_duration:.1f}s)")
                
                predictions = self.predict_emotion(temp_file, show_waveform=False, show_probabilities=False)
                segment_results[f"segment_{i+1}"] = predictions

                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            self.analyze_temporal_emotions(segment_results)
            return segment_results
            
        except Exception as e:
            print(f"Error in real-time prediction: {str(e)}")
            return None
    
    def analyze_temporal_emotions(self, segment_results):
        print(f"\nTEMPORAL EMOTION ANALYSIS")
        print("="*40)
        
        for model_name in self.models.keys():
            print(f"\n{model_name}:")
            emotions_sequence = []
            
            for segment_name, predictions in segment_results.items():
                if predictions and model_name in predictions and predictions[model_name]:
                    emotion = predictions[model_name]['emotion']
                    confidence = predictions[model_name]['confidence']
                    emotions_sequence.append(f"{emotion}({confidence:.2f})")
            
            print(f"   Timeline: {' → '.join(emotions_sequence)}")

def test_external_audio():
    MODEL_PATH = "/home/jhan/Documentos/modelo"
    
    tester = ExternalAudioTester(MODEL_PATH)
    tester.load_models()
    
    external_file = "/home/jhan/Documentos/proyectoTesis/tesis/Clasificacion-Emociones-Ravdess/audiosDePrueba/empresarioEnfadado.wav"
    
    if os.path.exists(external_file):
        predictions = tester.compare_models_on_file(external_file)
        tester.real_time_predict(external_file, segment_duration=3.0)
    else:
        print("Please provide a valid audio file path")

def test_folder():
    MODEL_PATH = "/home/jhan/Documentos/modelo"
    AUDIO_FOLDER = "/path/to/your/audio/folder"
    
    tester = ExternalAudioTester(MODEL_PATH)
    tester.load_models()
    
    batch_results = tester.batch_predict(AUDIO_FOLDER)
    return batch_results

def test_microphone_recording():
    try:
        import sounddevice as sd
        import scipy.io.wavfile as wavfile
        
        MODEL_PATH = "/home/jhan/Documentos/modelo"
        
        tester = ExternalAudioTester(MODEL_PATH)
        tester.load_models()
        
        duration = 5
        sample_rate = 22050
        
        print(f"Recording for {duration} seconds...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        
        temp_file = "recorded_audio.wav"
        wavfile.write(temp_file, sample_rate, audio_data)
        
        predictions = tester.compare_models_on_file(temp_file)
        
        os.remove(temp_file)
        return predictions
        
    except ImportError:
        print("sounddevice not installed. Install with: pip install sounddevice")
        return None

if __name__ == "__main__":
    test_external_audio()