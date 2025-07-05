import os
import time
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelBinarizer
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import psutil
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, model_path, dataset_path):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.emotion_labels = {
            '01': 'Neutral', '02': 'Calm', '03': 'Happy', '04': 'Sad',
            '05': 'Angry', '06': 'Fearful', '07': 'Disgust', '08': 'Surprised'
        }
        self.models = {}
        self.results = {}
        
    def load_models(self):
        """Carga todos los modelos entrenados"""
        print("Cargando modelos...")

        dt_path = os.path.join(self.model_path, "decision_tree_40mfccs.joblib")
        rf_path = os.path.join(self.model_path, "random_forest_40mfccs.joblib")
        cnn_path = os.path.join(self.model_path, "Emotion_Voice_Detection_Model_CNN.h5")
        
        try:
            if os.path.exists(dt_path):
                self.models['Decision Tree'] = joblib.load(dt_path)
                print("Decision Tree cargado")
            else:
                print("Decision Tree no encontrado")
                
            if os.path.exists(rf_path):
                self.models['Random Forest'] = joblib.load(rf_path)
                print("Random Forest cargado")
            else:
                print("Random Forest no encontrado")
                
            if os.path.exists(cnn_path):
                self.models['CNN'] = load_model(cnn_path)
                print("CNN cargado")
            else:
                print("CNN no encontrado")
                
        except Exception as e:
            print(f"Error cargando modelos: {str(e)}")
    
    def extract_features(self, file_path, max_length=130):
        """Extrae características de un archivo de audio"""
        try:
            X, sr = librosa.load(file_path, sr=22050, duration=3)
            mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40)
            delta = librosa.feature.delta(mfccs)
            combined = np.vstack([mfccs, delta]).T

            padded = pad_sequences([combined], maxlen=max_length, 
                                 dtype="float32", padding="post", truncating="post")
            
            return padded[0], padded[0].flatten()
        except Exception as e:
            print(f"Error procesando {file_path}: {str(e)}")
            return None, None
    
    def load_test_data(self, test_size=0.2):
        """Carga datos de prueba del dataset"""
        print("Cargando datos de prueba...")
        
        features_2d = []
        features_flat = []
        labels = []
        file_paths = []
        
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if not file.endswith(".wav"):
                    continue
                    
                file_path = os.path.join(root, file)
                feat_2d, feat_flat = self.extract_features(file_path)
                
                if feat_2d is not None:
                    features_2d.append(feat_2d)
                    features_flat.append(feat_flat)
                    labels.append(file.split("-")[2])
                    file_paths.append(file_path)

        X_2d = np.array(features_2d)
        X_flat = np.array(features_flat)
        y = np.array(labels)

        n_test = int(len(X_2d) * test_size)
        indices = np.random.choice(len(X_2d), n_test, replace=False)
        
        return (X_2d[indices], X_flat[indices], y[indices], 
                [file_paths[i] for i in indices])
    
    def measure_performance(self, model, X, y, model_name):
        """Mide el rendimiento de un modelo"""
        print(f"Evaluando {model_name}...")

        start_time = time.time()
        memory_before = psutil.virtual_memory().used / 1024 / 1024  # MB

        if model_name == 'CNN':
            X_cnn = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
            predictions = model.predict(X_cnn, verbose=0)
            y_pred = np.argmax(predictions, axis=1)
            y_pred_proba = predictions
        else:
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)
        
        end_time = time.time()
        memory_after = psutil.virtual_memory().used / 1024 / 1024  # MB

        inference_time = end_time - start_time
        memory_usage = memory_after - memory_before

        if isinstance(y[0], str):
            label_map = {str(i+1).zfill(2): i for i in range(8)}
            y_numeric = np.array([label_map[label] for label in y])
        else:
            y_numeric = y
            
        if isinstance(y_pred[0], str):
            y_pred_numeric = np.array([label_map[label] for label in y_pred])
        else:
            y_pred_numeric = y_pred

        accuracy = accuracy_score(y_numeric, y_pred_numeric)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_numeric, y_pred_numeric, average='weighted')

        cm = confusion_matrix(y_numeric, y_pred_numeric)

        try:
            lb = LabelBinarizer()
            y_binary = lb.fit_transform(y_numeric)
            if y_binary.shape[1] == 1:  # Caso binario
                y_binary = np.hstack([1-y_binary, y_binary])
            auc_score = roc_auc_score(y_binary, y_pred_proba, multi_class='ovr')
        except:
            auc_score = 0.0
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'inference_time': inference_time,
            'avg_inference_per_sample': inference_time / len(X),
            'memory_usage_mb': memory_usage,
            'confusion_matrix': cm,
            'y_true': y_numeric,
            'y_pred': y_pred_numeric,
            'y_pred_proba': y_pred_proba
        }
    
    def evaluate_all_models(self):
        """Evalúa todos los modelos cargados"""
        print("Iniciando evaluación completa...")

        X_2d, X_flat, y, file_paths = self.load_test_data()
        
        print(f"Datos de prueba: {len(X_2d)} muestras")
        print(f"Distribución de clases: {np.bincount(y.astype(int))}")

        for model_name, model in self.models.items():
            if model_name == 'CNN':
                result = self.measure_performance(model, X_2d, y, model_name)
            else:
                result = self.measure_performance(model, X_flat, y, model_name)
            
            self.results[model_name] = result
    
    def generate_comparison_report(self):
        """Genera un reporte comparativo de todos los modelos"""
        if not self.results:
            print("No hay resultados para comparar")
            return
        
        print("\n" + "="*80)
        print("REPORTE DE COMPARACIÓN DE MODELOS")
        print("="*80)

        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Modelo': model_name,
                'Precisión (%)': f"{result['accuracy']*100:.2f}",
                'F1-Score': f"{result['f1_score']:.3f}",
                'AUC': f"{result['auc_score']:.3f}",
                'Tiempo/Muestra (ms)': f"{result['avg_inference_per_sample']*1000:.2f}",
                'Memoria (MB)': f"{result['memory_usage_mb']:.2f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print("\nTABLA COMPARATIVA:")
        print(df_comparison.to_string(index=False))

        best_accuracy = max(self.results.keys(), 
                           key=lambda x: self.results[x]['accuracy'])
        best_speed = min(self.results.keys(), 
                        key=lambda x: self.results[x]['avg_inference_per_sample'])
        best_f1 = max(self.results.keys(), 
                     key=lambda x: self.results[x]['f1_score'])
        
        print(f"\nMEJORES MODELOS:")
        print(f"   • Precisión: {best_accuracy} ({self.results[best_accuracy]['accuracy']*100:.2f}%)")
        print(f"   • Velocidad: {best_speed} ({self.results[best_speed]['avg_inference_per_sample']*1000:.2f} ms/muestra)")
        print(f"   • F1-Score: {best_f1} ({self.results[best_f1]['f1_score']:.3f})")
        
        return df_comparison
    
    def plot_results(self):
        """Genera gráficos comparativos"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        accuracies = [r['accuracy']*100 for r in self.results.values()]
        speeds = [r['avg_inference_per_sample']*1000 for r in self.results.values()]
        model_names = list(self.results.keys())
        
        axes[0,0].scatter(speeds, accuracies, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            axes[0,0].annotate(name, (speeds[i], accuracies[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[0,0].set_xlabel('Velocidad (ms/muestra)')
        axes[0,0].set_ylabel('Precisión (%)')
        axes[0,0].set_title('Precisión vs Velocidad')
        axes[0,0].grid(True, alpha=0.3)

        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in model_names]
            axes[0,1].bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        axes[0,1].set_xlabel('Modelos')
        axes[0,1].set_ylabel('Puntuación')
        axes[0,1].set_title('Métricas por Modelo')
        axes[0,1].set_xticks(x + width * 1.5)
        axes[0,1].set_xticklabels(model_names)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        times = [r['avg_inference_per_sample']*1000 for r in self.results.values()]
        axes[1,0].bar(model_names, times, color=['skyblue', 'lightgreen', 'salmon'])
        axes[1,0].set_ylabel('Tiempo (ms/muestra)')
        axes[1,0].set_title('Tiempo de Inferencia por Modelo')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        memory = [r['memory_usage_mb'] for r in self.results.values()]
        axes[1,1].bar(model_names, memory, color=['lightcoral', 'lightyellow', 'lightblue'])
        axes[1,1].set_ylabel('Memoria (MB)')
        axes[1,1].set_title('Uso de Memoria por Modelo')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices(self):
        """Muestra las matrices de confusión de todos los modelos"""
        n_models = len(self.results)
        if n_models == 0:
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        emotion_names = list(self.emotion_labels.values())
        
        for i, (model_name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], 
                       xticklabels=emotion_names, yticklabels=emotion_names)
            axes[i].set_title(f'Matriz de Confusión - {model_name}')
            axes[i].set_xlabel('Predicción')
            axes[i].set_ylabel('Verdadero')
        
        plt.tight_layout()
        plt.show()
    
    def test_single_audio(self, audio_path):
        """Prueba un archivo de audio individual con todos los modelos"""
        print(f"Probando archivo: {os.path.basename(audio_path)}")
 
        feat_2d, feat_flat = self.extract_features(audio_path)
        if feat_2d is None:
            print("Error procesando el archivo")
            return

        true_emotion = audio_path.split("-")[2]
        true_emotion_name = self.emotion_labels.get(true_emotion, "Desconocido")
        
        print(f"Emoción verdadera: {true_emotion_name}")
        print("\nPredicciones:")
        
        for model_name, model in self.models.items():
            start_time = time.time()
            
            if model_name == 'CNN':
                X_input = feat_2d.reshape(1, feat_2d.shape[0], feat_2d.shape[1], 1)
                prediction = model.predict(X_input, verbose=0)
                pred_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
            else:
                X_input = feat_flat.reshape(1, -1)
                pred_class = model.predict(X_input)[0]
                confidence = np.max(model.predict_proba(X_input)[0])
            
            inference_time = time.time() - start_time

            if isinstance(pred_class, str):
                pred_emotion_name = self.emotion_labels.get(pred_class, "Desconocido")
            else:
                pred_emotion_key = str(pred_class + 1).zfill(2)
                pred_emotion_name = self.emotion_labels.get(pred_emotion_key, "Desconocido")

            is_correct = "Bueno" if pred_emotion_name == true_emotion_name else "No Bueno"
            
            print(f"   {model_name:15} → {pred_emotion_name:10} "
                  f"({confidence:.3f}) {is_correct} [{inference_time*1000:.2f}ms]")

def run_evaluation():
    """Ejecuta la evaluación completa del sistema"""

    DATASET_PATH = "/home/jhan/Documentos/archive"
    MODEL_PATH = "/home/jhan/Documentos/modelo"

    evaluator = ModelEvaluator(MODEL_PATH, DATASET_PATH)

    evaluator.load_models()
    
    if not evaluator.models:
        print("No se pudieron cargar los modelos")
        return

    evaluator.evaluate_all_models()

    df_comparison = evaluator.generate_comparison_report()

    evaluator.plot_results()
    evaluator.plot_confusion_matrices()

    sample_audio = os.path.join(DATASET_PATH, "Actor_01", "03-01-01-01-01-01-01.wav")
    if os.path.exists(sample_audio):
        print("\n" + "="*50)
        print("PRUEBA CON ARCHIVO INDIVIDUAL")
        print("="*50)
        evaluator.test_single_audio(sample_audio)
    
    return evaluator, df_comparison

if __name__ == "__main__":
    evaluator, comparison_df = run_evaluation()