import os
import numpy as np
import pandas as pd
from pathlib import Path

class EmotionTester:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.emotion_labels = {
            '01': 'Neutral',    '02': 'Calm',     '03': 'Happy',    '04': 'Sad',
            '05': 'Angry',      '06': 'Fearful',  '07': 'Disgust',  '08': 'Surprised'
        }
        self.emotion_examples = {
            'Neutral': '03-01-01-01-01-01-01.wav',
            'Calm': '03-01-02-01-01-01-01.wav',  
            'Happy': '03-01-03-01-01-01-01.wav',
            'Sad': '03-01-04-01-01-01-01.wav',
            'Angry': '03-01-05-01-01-01-01.wav',
            'Fearful': '03-01-06-01-01-01-01.wav',
            'Disgust': '03-01-07-01-01-01-01.wav',
            'Surprised': '03-01-08-01-01-01-01.wav'
        }
    
    def test_emotion_examples(self, dataset_path, actor_folder="Actor_01"):
        """Prueba un ejemplo de cada emoción"""
        print("PRUEBA DE TODAS LAS EMOCIONES")
        print("="*60)
        
        results = {}
        
        for emotion_name, filename in self.emotion_examples.items():
            print(f"\nProbando: {emotion_name.upper()}")
            print("-" * 40)

            audio_path = os.path.join(dataset_path, actor_folder, filename)
            
            if not os.path.exists(audio_path):
                print(f"Archivo no encontrado: {filename}")
                continue

            emotion_results = self.test_single_emotion(audio_path)
            results[emotion_name] = emotion_results
            
        return results
    
    def test_single_emotion(self, audio_path):
        """Prueba un archivo específico con todos los modelos"""
        feat_2d, feat_flat = self.evaluator.extract_features(audio_path)
        if feat_2d is None:
            return None

        true_emotion = audio_path.split("-")[2]
        true_emotion_name = self.emotion_labels.get(true_emotion, "Desconocido")
        
        print(f"Emoción verdadera: {true_emotion_name}")
        
        results = {}
        
        for model_name, model in self.evaluator.models.items():
            try:
                if model_name == 'CNN':
                    X_input = feat_2d.reshape(1, feat_2d.shape[0], feat_2d.shape[1], 1)
                    prediction = model.predict(X_input, verbose=0)
                    pred_class = np.argmax(prediction[0])
                    confidence = np.max(prediction[0])
                    all_probs = prediction[0]
                else:
                    X_input = feat_flat.reshape(1, -1)
                    pred_class = model.predict(X_input)[0]
                    proba = model.predict_proba(X_input)[0]
                    confidence = np.max(proba)
                    all_probs = proba
                
                if isinstance(pred_class, str):
                    pred_emotion_name = self.emotion_labels.get(pred_class, "Desconocido")
                else:
                    pred_emotion_key = str(pred_class + 1).zfill(2)
                    pred_emotion_name = self.emotion_labels.get(pred_emotion_key, "Desconocido")
                
                is_correct = pred_emotion_name == true_emotion_name
                accuracy_symbol = "Bueno" if is_correct else "No Bueno"
                
                print(f"   {model_name:15} → {pred_emotion_name:10} "
                      f"({confidence:.3f}) {accuracy_symbol}")
                
                results[model_name] = {
                    'predicted': pred_emotion_name,
                    'confidence': confidence,
                    'correct': is_correct,
                    'probabilities': all_probs
                }
                
            except Exception as e:
                print(f"   {model_name:15} → Error: {str(e)}")
                results[model_name] = None
        
        return results
    
    def test_multiple_actors_same_emotion(self, dataset_path, emotion_code, num_actors=5):
        """Prueba la misma emoción con diferentes actores"""
        emotion_name = self.emotion_labels.get(emotion_code, "Desconocido")
        
        print(f"\nPRUEBA: {emotion_name.upper()} CON MÚLTIPLES ACTORES")
        print("="*60)
        
        results = {}
        
        for actor_num in range(1, num_actors + 1):
            actor_folder = f"Actor_{actor_num:02d}"
            
            actor_path = os.path.join(dataset_path, actor_folder)
            if not os.path.exists(actor_path):
                continue

            for filename in os.listdir(actor_path):
                if filename.endswith('.wav') and filename.split('-')[2] == emotion_code:
                    audio_path = os.path.join(actor_path, filename)
                    
                    print(f"\n{actor_folder} - {filename}")
                    print("-" * 30)
                    
                    actor_results = self.test_single_emotion(audio_path)
                    results[f"{actor_folder}"] = actor_results
                    break
        
        return results
    
    def analyze_emotion_confusion(self, results):
        """Analiza qué emociones se confunden más"""
        print(f"\nANÁLISIS DE CONFUSIÓN POR EMOCIÓN")
        print("="*50)
        
        for emotion, emotion_results in results.items():
            if emotion_results is None:
                continue
                
            print(f"\n{emotion.upper()}:")
            
            for model_name, model_results in emotion_results.items():
                if model_results is None:
                    continue
                    
                if model_results['correct']:
                    print(f"   {model_name}: Correcto ({model_results['confidence']:.3f})")
                else:
                    print(f"   {model_name}: Predijo '{model_results['predicted']}' "
                          f"({model_results['confidence']:.3f})")
    
    def get_emotion_statistics(self, dataset_path):
        """Obtiene estadísticas por emoción en todo el dataset"""
        print("\nESTADÍSTICAS POR EMOCIÓN EN EL DATASET")
        print("="*50)
        
        emotion_counts = {}
        
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    emotion_code = file.split('-')[2]
                    emotion_name = self.emotion_labels.get(emotion_code, "Desconocido")
                    
                    if emotion_name not in emotion_counts:
                        emotion_counts[emotion_name] = 0
                    emotion_counts[emotion_name] += 1

        total_files = sum(emotion_counts.values())
        
        for emotion, count in sorted(emotion_counts.items()):
            percentage = (count / total_files) * 100
            print(f"   {emotion:10} → {count:3d} archivos ({percentage:.1f}%)")
        
        print(f"\nTotal: {total_files} archivos de audio")
        return emotion_counts

def run_emotion_tests(dataset_path, model_path):
    """Ejecuta pruebas específicas por emoción"""
    
    from implementacion import ModelEvaluator
    evaluator = ModelEvaluator(model_path, dataset_path)
    evaluator.load_models()
    
    emotion_tester = EmotionTester(evaluator)
    
    emotion_tester.get_emotion_statistics(dataset_path)

    all_results = emotion_tester.test_emotion_examples(dataset_path)

    emotion_tester.analyze_emotion_confusion(all_results)

    print("\n" + "="*60)
    angry_results = emotion_tester.test_multiple_actors_same_emotion(
        dataset_path, '05', num_actors=3)
    
    return emotion_tester, all_results

def test_specific_emotions():
    """Ejemplos de cómo probar emociones específicas"""
    
    DATASET_PATH = "/home/jhan/Documentos/archive"
    MODEL_PATH = "/home/jhan/Documentos/modelo"

    from implementacion import ModelEvaluator
    evaluator = ModelEvaluator(MODEL_PATH, DATASET_PATH)
    evaluator.load_models()

    emotion_files = {
        'Happy': os.path.join(DATASET_PATH, "Actor_01", "03-01-03-01-01-01-01.wav"),
        'Sad': os.path.join(DATASET_PATH, "Actor_01", "03-01-04-01-01-01-01.wav"),
        'Angry': os.path.join(DATASET_PATH, "Actor_01", "03-01-05-01-01-01-01.wav"),
        'Fearful': os.path.join(DATASET_PATH, "Actor_02", "03-01-06-01-01-01-01.wav"),
        'Surprised': os.path.join(DATASET_PATH, "Actor_03", "03-01-08-01-01-01-01.wav")
    }
    
    print("PRUEBAS ESPECÍFICAS POR EMOCIÓN")
    print("="*50)
    
    for emotion_name, file_path in emotion_files.items():
        if os.path.exists(file_path):
            print(f"\n🎵 Probando: {emotion_name}")
            print("-" * 30)
            evaluator.test_single_audio(file_path)
        else:
            print(f"Archivo no encontrado para {emotion_name}: {file_path}")

if __name__ == "__main__":
    DATASET_PATH = "/home/jhan/Documentos/archive"
    MODEL_PATH = "/home/jhan/Documentos/modelo"
    
    emotion_tester, results = run_emotion_tests(DATASET_PATH, MODEL_PATH)
