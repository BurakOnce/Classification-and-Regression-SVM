import scipy.io
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.metrics import make_scorer
import numpy as np

def smape_score(y_true, y_pred):
    return 200 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))


# Veri Setleri
c3_data = scipy.io.loadmat('C:/Users/burak/OneDrive/Masaüstü/ÖRÜNTÜ2024_Proje/C3/wifi_localization.mat')
feat_c3 = c3_data['feat']
lbl_c3 = c3_data['lbl']

print("C3 Veri Seti:")
print("feat_c3 shape:", feat_c3.shape)
print("lbl_c3 shape:", lbl_c3.shape)


# KFold nesnesi
kfold = KFold(n_splits=3, shuffle=True, random_state=42)

# SVM Sınıflandırma Modeli
svm_model_class = SVC(kernel='rbf')

svm_scores_class = {
    'accuracy': cross_val_score(svm_model_class, feat_c3, lbl_c3.ravel(), cv=kfold, scoring='accuracy'),
    'f1_score': cross_val_score(svm_model_class, feat_c3, lbl_c3.ravel(), cv=kfold, scoring='f1_weighted')
}

# SVM Regresyon Modeli
svm_model_reg = SVR(kernel='rbf')

# SMAPE ve MAE için scorer
smape_scorer = make_scorer(smape_score, greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

svm_scores_reg = {
    'mae': -cross_val_score(svm_model_reg, feat_c3, lbl_c3.ravel(), cv=kfold, scoring=mae_scorer),
    'smape': -cross_val_score(svm_model_reg, feat_c3, lbl_c3.ravel(), cv=kfold, scoring=smape_scorer)
}

print("\nSVM Classification Scores:")
print("Accuracy:", np.mean(svm_scores_class['accuracy']))
print("F1 Score:", np.mean(svm_scores_class['f1_score']))

print("\nSVM Regression Scores:")
print("MAE:", np.mean(svm_scores_reg['mae']))
print("SMAPE:", np.mean(svm_scores_reg['smape']))