"""
shap_analysis.py
-----------------
Etap 4: Wyjasnialna sztuczna inteligencja (XAI) za pomoca SHAP.

Wczytuje najlepszy wytrenowany model (models/best_model.pkl) i generuje:
1. Globalny wykres SHAP (summary/beeswarm) - jak kazda cecha wplywa na cene
   w calym zbiorze danych.
2. Wykres waznosci cech (bar plot) - srednia bezwzgledna wartosc SHAP.
3. Lokalny wykres typu waterfall dla pojedynczego przykladu - pokazuje,
   jak poszczegolne cechy tego konkretnego mieszkania zmieniaja cene
   wzgledem wartosci bazowej (oczekiwanej).
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

DATA_PATH = "data/cleaned_apartments.csv"
MODEL_PATH = "models/best_model.pkl"

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["model"]
model_name = bundle["model_name"]
numeric_features = bundle["numeric_features"]
categorical_features = bundle["categorical_features"]

print(f"Analiza SHAP dla modelu: {model_name}")

df = pd.read_csv(DATA_PATH)
X = df[numeric_features + categorical_features]

# probka do analizy SHAP (dla szybkosci obliczen na 1 rdzeniu CPU)
X_sample = X.sample(n=min(120, len(X)), random_state=42).reset_index(drop=True)

prep = pipeline.named_steps["prep"]
model = pipeline.named_steps["model"]

X_transformed = prep.transform(X_sample)
feature_names = prep.get_feature_names_out()
# czytelniejsze nazwy cech (bez prefiksow "num__"/"cat__")
feature_names = [f.replace("num__", "").replace("cat__city_", "city=") for f in feature_names]

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_transformed)
shap_values.feature_names = feature_names

# 1. Globalny wykres SHAP (beeswarm)
plt.figure()
shap.summary_plot(shap_values, show=False)
plt.title(f"SHAP - globalny wplyw cech na cene ({model_name})")
plt.tight_layout()
plt.savefig("figures/06_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Waznosc cech (bar plot)
plt.figure()
shap.summary_plot(shap_values, plot_type="bar", show=False)
plt.title(f"SHAP - waznosc cech ({model_name})")
plt.tight_layout()
plt.savefig("figures/07_shap_importance_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Lokalne wyjasnienie (waterfall) dla przykladowego mieszkania
example_idx = 0
plt.figure()
shap.plots.waterfall(shap_values[example_idx], show=False)
plt.title("SHAP waterfall - wyjasnienie dla pojedynczego mieszkania")
plt.tight_layout()
plt.savefig("figures/08_shap_waterfall_example.png", dpi=150, bbox_inches="tight")
plt.close()

print("Przykladowe mieszkanie uzyte do wyjasnienia lokalnego:")
print(X_sample.iloc[example_idx])
print(f"Przewidziana cena: {model.predict(X_transformed[example_idx:example_idx+1])[0]:,.0f} PLN")
expected_value = explainer.expected_value
if hasattr(expected_value, "__len__"):
    expected_value = expected_value[0]
print(f"Wartosc bazowa (oczekiwana) modelu: {expected_value:,.0f} PLN")

print("\nZapisano 3 wykresy SHAP w folderze figures/:")
print(" - 06_shap_summary.png (globalny)")
print(" - 07_shap_importance_bar.png (waznosc cech)")
print(" - 08_shap_waterfall_example.png (lokalny przyklad)")
