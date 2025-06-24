"""
Bitirme Projesi - Makine Öğrenmesi ile Veri Analizi
Öğrenci: Elif Doylan
Öğrenci No: 20360859003
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any
import logging
from datetime import datetime
from itertools import product

# Scikit-learn imports
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# XGBoost import
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

import warnings
warnings.filterwarnings("ignore")

# Matplotlib Türkçe karakter desteği
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Lucida Grande']

def folded_training(model, X_train, y_train):
    """5-fold cross validation training"""
    k = 5 
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mae_list, rmse_list, r2_list = [], [], []
    
    if hasattr(X_train, "values"):
        X_train = X_train.values
    if hasattr(y_train, "values"):
        y_train = y_train.values

    y_pred_all = np.zeros(len(y_train))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_pred_all[val_idx] = y_pred 

        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
        print(f"Fold {fold+1}: MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.2f}")
    return mae_list, rmse_list, r2_list, y_pred_all 

def model_search(model_class, param_grid, X_train, y_train):
    """Grid search ile en iyi parametreleri bul"""
    model_name = model_class.__name__
    best_score = float("inf")
    best_model = None
    best_result = None 
    best_params = None 
    
    print(f"Hyperparameter search for: {model_class.__name__}")
    keys = list(param_grid.keys())
    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))
        if "Polynomial" in model_class.__name__:
            model = make_pipeline(PolynomialFeatures(**params), Ridge())
        else:
            model = model_class(**params)

        mae_list, rmse_list, r2_list, y_pred_all = folded_training(model, X_train, y_train)
        score_mae = np.mean(mae_list)
        
        print(f"Params: {params} | MAE: {score_mae:.3f}\n")

        if score_mae < best_score:
            best_score = score_mae
            best_model = model
            best_result = (mae_list, rmse_list, r2_list, y_pred_all)
            best_params = params
    print(f"\n=== Best params: {best_params} ===")
    show_results_CV(model_name, best_result[0], best_result[1], best_result[2], best_result[3], y_train)
    return best_model, best_result

def show_results_CV(model_name, mae_list, rmse_list, r2_list, y_pred_all, y_train):
    """Cross validation sonuçlarını göster"""
    print(f"\nOrtalama MAE= {np.mean(mae_list):.2f}")
    print(f"Ortalama RMSE= {np.mean(rmse_list):.2f}")
    print(f"Ortalama R2= {np.mean(r2_list):.2f}")

def train_and_test_final_model(model, X_train, y_train, X_test, y_test):
    """Final modeli test et"""
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"Test MAE = {test_mae:.2f}")
    print(f"Test RMSE = {test_rmse:.2f}")
    print(f"Test R2 = {test_r2:.2f}")
    return model, y_test_pred, test_mae, test_rmse, test_r2

def save_test_predictions(model, X_test, y_test, y_pred, model_name, save_path='test_predictions'):
    """Test tahminlerini kaydet"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df = pd.DataFrame({
        'Actual_Values': y_test,
        f'Predictions_{model_name}': y_pred,
        f'Error_{model_name}': y_test - y_pred
    })
    output_path = f'{save_path}_{model_name}_{timestamp}.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    return results_df

class DataAnalyzer:
    """Ana veri analiz sınıfı"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Veri analizci başlatıcı
        
        Args:
            config: Konfigürasyon parametreleri
        """
        self.config = config or self._default_config()
        self.setup_logging()
        self.data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.results = []
        
    def _default_config(self) -> Dict[str, Any]:
        """Varsayılan konfigürasyon"""
        return {
            'data_file': 'latest_data.xlsx',
            'test_size': 0.15,
            'random_state': 42,
            'cv_folds': 5,
            'output_dir': 'outputs',
            'save_plots': True,
            'plot_dpi': 300,
            'hub_threshold_high': 5.5,
            'hub_threshold_low': 3.5,
            'target_keywords': ['LG35', 'target'],
            'hub_keywords': ['hub', 'Hub'],
            'ensemble_weights': [2, 3, 1]  # GB, XGB, RF
        }
    
    def setup_logging(self):
        """Logging ayarları"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('analysis.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def ensure_output_dir(self):
        """Çıktı klasörünü oluştur"""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    def load_data(self) -> pd.DataFrame:
        """Veriyi yükle ve temel bilgileri göster"""
        try:
            data_file = self.config['data_file']
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Veri dosyası bulunamadı: {data_file}")
            
            self.logger.info(f"Veri dosyası yükleniyor: {data_file}")
            self.data = pd.read_excel(data_file)
            
            self.logger.info(f"Orijinal veri boyutu: {self.data.shape}")
            self.logger.info(f"Veri kolonları: {len(self.data.columns)} adet")
            self.logger.info(f"Toplam gözlem sayısı: {len(self.data)}")
            
            return self.data
            
        except Exception as e:
            self.logger.error(f"Veri yükleme hatası: {e}")
            raise
    
    def preprocess_data(self) -> pd.DataFrame:
        """Veriyi ön işleme tabi tut"""
        if self.data is None:
            raise ValueError("Önce veri yüklenmelidir")
        
        self.logger.info("Veri ön işleme başlıyor...")
        
        # Kopya oluştur
        processed_data = self.data.copy()
        
        # Kolon adlarındaki boşlukları temizle
        processed_data.columns = processed_data.columns.str.strip()
        
        # Eksik değerleri kontrol et ve doldur
        missing_before = processed_data.isnull().sum().sum()
        self.logger.info(f"Toplam eksik değer sayısı: {missing_before}")
        
        # Numerik kolonlar için ortalama ile doldur
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if processed_data[col].isnull().any():
                mean_val = processed_data[col].mean()
                processed_data[col].fillna(mean_val, inplace=True)
                self.logger.info(f"'{col}' kolonu ortalama ile dolduruldu: {mean_val:.2f}")
        
        # Kategorik kolonlar için mod ile doldur
        categorical_columns = processed_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if processed_data[col].isnull().any():
                mode_val = processed_data[col].mode().iloc[0] if not processed_data[col].mode().empty else "Unknown"
                processed_data[col].fillna(mode_val, inplace=True)
                self.logger.info(f"'{col}' kolonu mod ile dolduruldu: {mode_val}")
        
        missing_after = processed_data.isnull().sum().sum()
        self.logger.info(f"Doldurma sonrası eksik değer sayısı: {missing_after}")
        
        self.processed_data = processed_data
        return processed_data
    
    def encode_categorical_features(self) -> pd.DataFrame:
        """Kategorik değişkenleri encode et"""
        if self.processed_data is None:
            raise ValueError("Önce veri ön işleme yapılmalıdır")
        
        categorical_columns = self.processed_data.select_dtypes(include=['object']).columns.tolist()
        
        if not categorical_columns:
            self.logger.info("Encode edilecek kategorik kolon bulunamadı")
            return self.processed_data
        
        self.logger.info(f"Kategorik kolonlar encode ediliyor: {categorical_columns}")
        
        encoded_data = self.processed_data.copy()
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            encoded_data[col] = le.fit_transform(encoded_data[col])
            label_encoders[col] = le
            self.logger.info(f"'{col}' kolonu encode edildi - {len(le.classes_)} benzersiz değer")
        
        self.processed_data = encoded_data
        self.label_encoders = label_encoders
        return encoded_data
    
    def find_target_column(self) -> str:
        """Hedef kolonu otomatik olarak bul"""
        target_keywords = self.config['target_keywords']
        
        for keyword in target_keywords:
            candidates = [col for col in self.processed_data.columns if keyword in col]
            if candidates:
                target_col = candidates[0]
                self.logger.info(f"Hedef kolon bulundu: '{target_col}'")
                return target_col
        
        # Eğer bulunamazsa, kullanıcıdan seçim yapmasını iste
        self.logger.warning("Otomatik hedef kolon bulunamadı")
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 1:
            return numeric_cols[0]
        
        print("\nMevcut numerik kolonlar:")
        for i, col in enumerate(numeric_cols):
            print(f"{i+1}. {col}")
        
        while True:
            try:
                choice = int(input("\nHedef kolon numarasını seçin: ")) - 1
                if 0 <= choice < len(numeric_cols):
                    return numeric_cols[choice]
                else:
                    print("Geçersiz seçim, tekrar deneyin.")
            except ValueError:
                print("Lütfen geçerli bir sayı girin.")
    
    def split_data(self, target_column: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Veriyi eğitim ve test setlerine böl"""
        # Hub kolonu otomatik olarak bul
        hub_keywords = self.config['hub_keywords']
        hub_column = None
        
        for keyword in hub_keywords:
            hub_candidates = [col for col in self.processed_data.columns if keyword.lower() in col.lower()]
            if hub_candidates:
                hub_column = hub_candidates[0]
                self.logger.info(f"Hub kolonu bulundu: '{hub_column}'")
                break
        
        if hub_column is None:
            self.logger.warning("Hub kolonu bulunamadı, rastgele bölme yapılıyor")
            X = self.processed_data.drop(columns=[target_column])
            y = self.processed_data[target_column]
            return train_test_split(X, y, test_size=self.config['test_size'], 
                                  random_state=self.config['random_state'])
        
        # Hub koşullarına göre böl
        high_threshold = self.config['hub_threshold_high']
        low_threshold = self.config['hub_threshold_low']
        
        mask_high = (self.processed_data[hub_column] > high_threshold)
        mask_low = (self.processed_data[hub_column] < low_threshold)
        test_mask = mask_high | mask_low
        
        df_test = self.processed_data[test_mask]
        df_train = self.processed_data[~test_mask]
        
        self.logger.info(f"Hub koşulları: >{high_threshold} veya <{low_threshold}")
        self.logger.info(f"Eğitim seti boyutu: {len(df_train)}")
        self.logger.info(f"Test seti boyutu: {len(df_test)}")
        
        X_train = df_train.drop(columns=[target_column])
        y_train = df_train[target_column]
        X_test = df_test.drop(columns=[target_column])
        y_test = df_test[target_column]
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self):
        """Özellikleri ölçeklendir"""
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.logger.info("Özellikler standardize edildi")
    
    def cross_validation(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """K-fold çapraz doğrulama"""
        kf = KFold(n_splits=self.config['cv_folds'], shuffle=True, 
                  random_state=self.config['random_state'])
        
        scores = {'mae': [], 'rmse': [], 'r2': []}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            scores['mae'].append(mae)
            scores['rmse'].append(rmse)
            scores['r2'].append(r2)
        
        return {
            'cv_mae_mean': np.mean(scores['mae']),
            'cv_mae_std': np.std(scores['mae']),
            'cv_rmse_mean': np.mean(scores['rmse']),
            'cv_rmse_std': np.std(scores['rmse']),
            'cv_r2_mean': np.mean(scores['r2']),
            'cv_r2_std': np.std(scores['r2'])
        }
    
    def train_and_evaluate_model(self, name: str, model) -> Dict[str, Any]:
        """Model eğit ve değerlendir"""
        self.logger.info(f"{name} modeli eğitiliyor...")
        
        # Çapraz doğrulama
        cv_results = self.cross_validation(model, self.X_train_scaled, self.y_train)
        
        # Test seti üzerinde değerlendirme
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        
        test_mae = mean_absolute_error(self.y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        test_r2 = r2_score(self.y_test, y_pred)
        
        results = {
            'model_name': name,
            'model': model,
            'predictions': y_pred,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            **cv_results
        }
        
        self.logger.info(f"{name} - CV MAE: {cv_results['cv_mae_mean']:.3f} ± {cv_results['cv_mae_std']:.3f}")
        self.logger.info(f"{name} - Test MAE: {test_mae:.3f}, R²: {test_r2:.3f}")
        
        return results
    
    def create_models(self) -> List[Tuple[str, Any]]:
        """Kullanılacak modelleri oluştur"""
        models = [
            ('Lineer Regresyon', LinearRegression()),
            ('Ridge Regresyon', Ridge(alpha=10.0)),
            ('Polinom+Ridge (derece=3)', make_pipeline(PolynomialFeatures(degree=3), Ridge())),
            ('Karar Ağacı', DecisionTreeRegressor(max_depth=None, min_samples_leaf=1)),
            ('Rastgele Orman', RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_leaf=5, random_state=42)),
            ('Gradyan Artırma', GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42))
        ]
        
        if XGBOOST_AVAILABLE:
            models.append(('XGBoost', XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)))
        
        return models
    
    def create_ensemble_model(self) -> VotingRegressor:
        """Topluluk modeli oluştur"""
        base_models = [
            ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=42))
        ]
        
        weights = [1, 1]  # GB, RF
        
        if XGBOOST_AVAILABLE:
            base_models.append(('xgb', XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)))
            weights = self.config['ensemble_weights']
        
        return VotingRegressor(estimators=base_models, weights=weights)
    
    def plot_correlation_heatmap(self):
        """Korelasyon ısı haritası"""
        plt.figure(figsize=(12, 8))
        numeric_data = self.processed_data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
                   square=True, linewidths=0.5)
        plt.title("Özellikler Arası Korelasyon Matrisi", fontsize=16, pad=20)
        plt.tight_layout()
        
        if self.config['save_plots']:
            output_dir = self.ensure_output_dir()
            plt.savefig(output_dir / 'correlation_heatmap.png', 
                       dpi=self.config['plot_dpi'], bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self):
        """Model karşılaştırması grafiği"""
        if not self.results:
            self.logger.warning("Sonuç bulunamadı, önce modelleri eğitin")
            return
        
        model_names = [r['model_name'] for r in self.results]
        cv_mae_values = [r['cv_mae_mean'] for r in self.results]
        test_mae_values = [r['test_mae'] for r in self.results]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, cv_mae_values, width, label='CV MAE', alpha=0.8)
        bars2 = ax.bar(x + width/2, test_mae_values, width, label='Test MAE', alpha=0.8)
        
        ax.set_xlabel('Modeller')
        ax.set_ylabel('Ortalama Mutlak Hata (MAE)')
        ax.set_title('Model Performans Karşılaştırması')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Değerleri çubukların üzerine yaz
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        plt.tight_layout()
        
        if self.config['save_plots']:
            output_dir = self.ensure_output_dir()
            plt.savefig(output_dir / 'model_comparison.png', 
                       dpi=self.config['plot_dpi'], bbox_inches='tight')
        
        plt.show()
    
    def plot_predictions_scatter(self):
        """Tahmin dağılım grafikleri"""
        if not self.results:
            return
        
        n_models = len(self.results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, result in enumerate(self.results):
            ax = axes[i]
            
            y_true = self.y_test
            y_pred = result['predictions']
            
            ax.scatter(y_true, y_pred, alpha=0.6, s=30)
            
            # Mükemmel tahmin çizgisi
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_xlabel('Gerçek Değerler')
            ax.set_ylabel('Tahmin Değerleri')
            ax.set_title(f"{result['model_name']}\nR² = {result['test_r2']:.3f}")
            ax.grid(True, alpha=0.3)
        
        # Kullanılmayan subplot'ları gizle
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if self.config['save_plots']:
            output_dir = self.ensure_output_dir()
            plt.savefig(output_dir / 'predictions_scatter.png', 
                       dpi=self.config['plot_dpi'], bbox_inches='tight')
        
        plt.show()
    
    def save_results(self):
        """Sonuçları CSV dosyasına kaydet"""
        if not self.results:
            return
        
        output_dir = self.ensure_output_dir()
        
        # Model sonuçları
        results_df = pd.DataFrame([
            {
                'Model': r['model_name'],
                'CV_MAE_Ortalama': r['cv_mae_mean'],
                'CV_MAE_Std': r['cv_mae_std'],
                'CV_RMSE_Ortalama': r['cv_rmse_mean'],
                'CV_R2_Ortalama': r['cv_r2_mean'],
                'Test_MAE': r['test_mae'],
                'Test_RMSE': r['test_rmse'],
                'Test_R2': r['test_r2']
            } for r in self.results
        ])
        
        results_df.to_csv(output_dir / 'model_results.csv', index=False)
        self.logger.info(f"Model sonuçları kaydedildi: {output_dir / 'model_results.csv'}")
        
        # Tahminler
        predictions_df = pd.DataFrame({'Gerçek_Değerler': self.y_test})
        for result in self.results:
            predictions_df[f"{result['model_name']}_Tahmin"] = result['predictions']
        
        predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
        self.logger.info(f"Tahminler kaydedildi: {output_dir / 'predictions.csv'}")
    
    def print_summary(self):
        """Sonuç özeti yazdır"""
        if not self.results:
            return
        
        print("\n" + "="*80)
        print("SONUÇ ÖZETİ")
        print("="*80)
        
        # En iyi modeli bul
        best_model = min(self.results, key=lambda x: x['test_mae'])
        
        for result in sorted(self.results, key=lambda x: x['test_mae']):
            status = " [EN İYİ]" if result == best_model else ""
            print(f"{result['model_name']:<25}: "
                  f"Test MAE={result['test_mae']:.3f}, "
                  f"Test R²={result['test_r2']:.3f}, "
                  f"CV MAE={result['cv_mae_mean']:.3f}±{result['cv_mae_std']:.3f}"
                  f"{status}")
        
        print(f"\nEn iyi model: {best_model['model_name']}")
        print(f"Veri seti boyutu: {len(self.processed_data)} (Eğitim: {len(self.y_train)}, Test: {len(self.y_test)})")
        print(f"Özellik sayısı: {self.X_train.shape[1]}")
    
    def run_complete_analysis(self):
        """Tam analizi çalıştır"""
        try:
            self.logger.info("Tam veri analizi başlıyor...")
            
            # 1. Veri yükleme
            self.load_data()
            
            # 2. Ön işleme
            self.preprocess_data()
            
            # 3. Korelasyon grafiği
            self.plot_correlation_heatmap()
            
            # 4. Kategorik encoding
            self.encode_categorical_features()
            
            # 5. Hedef kolon bulma
            target_column = self.find_target_column()
            
            # 6. Veri bölme
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(target_column)
            
            # 7. Ölçeklendirme
            self.scale_features()
            
            # 8. Hyperparameter tuning ile model eğitimi
            self.logger.info("Grid search ile en iyi parametreler bulunuyor...")
            
            # Linear Regression (parametre yok)
            print("\n" + "="*50)
            print("LINEAR REGRESSION")
            print("="*50)
            model = LinearRegression()
            model.fit(self.X_train_scaled, self.y_train)
            mae_list, rmse_list, r2_list, y_pred_all = folded_training(model, self.X_train_scaled, self.y_train)
            show_results_CV("Linear Regression", mae_list, rmse_list, r2_list, y_pred_all, self.y_train)
            
            # Ridge Regression
            print("\n" + "="*50)
            print("RIDGE REGRESSION")
            print("="*50)
            param_grid = {
                "alpha": [0.01, 0.1, 1.0, 10.0]
            }
            best_ridge, _ = model_search(Ridge, param_grid, self.X_train_scaled, self.y_train)
            
            # Polynomial Features + Ridge
            print("\n" + "="*50)
            print("POLYNOMIAL FEATURES + RIDGE")
            print("="*50)
            param_grid = {
                "degree": [2, 3]
            }
            best_poly, _ = model_search(PolynomialFeatures, param_grid, self.X_train_scaled, self.y_train)
            
            # Decision Tree
            print("\n" + "="*50)
            print("DECISION TREE")
            print("="*50)
            param_grid = {
                "max_depth": [2, 3, 5, 7, 10, 15, None],
                "min_samples_leaf": [1, 5, 10]
            }
            best_dt, _ = model_search(DecisionTreeRegressor, param_grid, self.X_train_scaled, self.y_train)
            
            # Random Forest
            print("\n" + "="*50)
            print("RANDOM FOREST")
            print("="*50)
            param_grid = {
                "n_estimators": [50, 100],
                "max_depth": [3, 5, 10, None],
                "min_samples_leaf": [1, 5]
            }
            best_rf, _ = model_search(RandomForestRegressor, param_grid, self.X_train_scaled, self.y_train)
            
            # Gradient Boosting
            print("\n" + "="*50)
            print("GRADIENT BOOSTING")
            print("="*50)
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [2, 3, 5],
                "learning_rate": [0.01, 0.1]
            }
            best_gb, _ = model_search(GradientBoostingRegressor, param_grid, self.X_train_scaled, self.y_train)
            
            # XGBoost (if available)
            if XGBOOST_AVAILABLE:
                print("\n" + "="*50)
                print("XGBOOST")
                print("="*50)
                param_grid = {
                    "n_estimators": [100, 200], 
                    "max_depth": [2, 3, 5],       
                    "learning_rate": [0.01, 0.1],
                }
                best_xgb, _ = model_search(XGBRegressor, param_grid, self.X_train_scaled, self.y_train)
            
            # 9. Test setinde final değerlendirme
            print("\n" + "="*70)
            print("FINAL TEST SET EVALUATION")
            print("="*70)
            
            models_to_test = [
                ("Linear Regression", model),
                ("Best Ridge", best_ridge),
                ("Best Polynomial+Ridge", best_poly),
                ("Best Decision Tree", best_dt),
                ("Best Random Forest", best_rf),
                ("Best Gradient Boosting", best_gb)
            ]
            
            if XGBOOST_AVAILABLE:
                models_to_test.append(("Best XGBoost", best_xgb))
            
            best_models = {}
            for model_name, model_obj in models_to_test:
                print(f"\n--- {model_name} ---")
                final_model, y_pred, test_mae, test_rmse, test_r2 = train_and_test_final_model(
                    model_obj, self.X_train_scaled, self.y_train, self.X_test_scaled, self.y_test
                )
                best_models[model_name] = {
                    'model': final_model,
                    'predictions': y_pred,
                    'test_mae': test_mae,
                    'test_rmse': test_rmse,
                    'test_r2': test_r2
                }
                
                # Tahminleri kaydet
                save_test_predictions(final_model, self.X_test, self.y_test, y_pred, model_name)
            
            # 10. En iyi modeli bul ve özet yazdır
            print("\n" + "="*70)
            print("FINAL MODEL COMPARISON")
            print("="*70)
            
            for model_name, results in best_models.items():
                print(f"{model_name:<25}: Test MAE={results['test_mae']:.3f}, Test R²={results['test_r2']:.3f}")
            
            # En iyi modeli bul
            best_model_name = min(best_models.keys(), key=lambda x: best_models[x]['test_mae'])
            print(f"\nEn iyi model: {best_model_name}")
            print(f"Test MAE: {best_models[best_model_name]['test_mae']:.3f}")
            print(f"Test R²: {best_models[best_model_name]['test_r2']:.3f}")
            
            self.best_models = best_models
            self.logger.info("Analiz başarıyla tamamlandı!")
            
        except Exception as e:
            self.logger.error(f"Analiz sırasında hata oluştu: {e}")
            raise

def main():
    """Ana fonksiyon"""
    print("Bitirme Projesi - Makine Öğrenmesi Veri Analizi")
    print("Öğrenci: Elif Doylan (20360859003)")
    print("=" * 50)
    
    # Konfigürasyon (isteğe bağlı olarak değiştirilebilir)
    config = {
        'data_file': 'latest_data.xlsx',
        'output_dir': 'outputs',
        'save_plots': True,
        'plot_dpi': 300,
        'cv_folds': 5,
        'random_state': 42
    }
    
    # Analiz nesnesini oluştur ve çalıştır
    analyzer = DataAnalyzer(config)
    analyzer.run_complete_analysis()
    
    print("\nAnaliz tamamlandı!")
    print(f"Sonuçlar '{config['output_dir']}' klasöründe kaydedildi.")

if __name__ == "__main__":
    main()
