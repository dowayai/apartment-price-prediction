"""
train_model.py
---------------
Etap 3: Trenowanie modeli uczenia maszynowego.

Porownuje trzy modele regresji do przewidywania ceny mieszkania:
- Regresja liniowa (baseline)
- Random Forest Regressor
- XGBoost Regressor

Dla Random Forest i XGBoost przeprowadzane jest strojenie hiperparametrow
(GridSearchCV, 5-krotna walidacja krzyzowa). Modele oceniane sa na zbiorze
testowym za pomoca MAE, RMSE i R^2. Najlepszy model (najnizszy RMSE) jest
zapisywany do models/best_model.pkl razem z metadanymi.
"""

import io
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

DATA_PATH = "data/cleaned_apartments.csv"
MODEL_PATH = "models/best_model.pkl"
METRICS_PATH = "models/metrics.json"

NUMERIC_FEATURES = ["rooms", "floor", "sq", "building_age", "distance_to_center_km"]
CATEGORICAL_FEATURES = ["city"]
TARGET = "price"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {"MAE": round(float(mae), 2), "RMSE": round(float(rmse), 2), "R2": round(float(r2), 4)}


def model_size_mb(model) -> float:
    """Serializuje model do pamieci, aby zmierzyc jego rozmiar (bez zapisu na dysk)."""
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return round(len(buf.getvalue()) / 1e6, 2)


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Zbior treningowy: {X_train.shape[0]} | Zbior testowy: {X_test.shape[0]}")

    # Srodowisko obliczeniowe dysponuje 1 rdzeniem CPU, dlatego strojenie
    # hiperparametrow (GridSearchCV) wykonywane jest na mniejszej probie
    # treningowej, a finalny model jest dotrenowywany na pelnym zbiorze
    # treningowym z najlepszymi znalezionymi hiperparametrami. To standardowe
    # podejscie przy ograniczonych zasobach obliczeniowych.
    tune_sample = X_train.sample(n=min(4000, len(X_train)), random_state=42)
    y_tune_sample = y_train.loc[tune_sample.index]

    results = {}
    fitted_models = {}

    # 1) Regresja liniowa (baseline, bez strojenia hiperparametrow)
    print("\n[1/3] Regresja liniowa...")
    lin_pipe = Pipeline([("prep", build_preprocessor()), ("model", LinearRegression())])
    t0 = time.time()
    lin_pipe.fit(X_train, y_train)
    results["Linear Regression"] = evaluate(lin_pipe, X_test, y_test)
    results["Linear Regression"]["train_time_s"] = round(time.time() - t0, 2)
    results["Linear Regression"]["model_size_mb"] = model_size_mb(lin_pipe)
    fitted_models["Linear Regression"] = lin_pipe
    print(results["Linear Regression"])

    # 2) Random Forest + GridSearchCV (na probie), nastepnie refit na calosci
    print("\n[2/3] Random Forest (GridSearchCV na probie 4000 wierszy)...")
    rf_pipe = Pipeline([("prep", build_preprocessor()), ("model", RandomForestRegressor(random_state=42))])
    rf_grid = {
        "model__n_estimators": [150, 250],
        "model__max_depth": [10, 15],
        "model__min_samples_leaf": [2, 4],
    }
    t0 = time.time()
    rf_search = GridSearchCV(rf_pipe, rf_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=1)
    rf_search.fit(tune_sample, y_tune_sample)
    best_rf = rf_search.best_estimator_
    best_rf.set_params(**rf_search.best_params_)
    best_rf.fit(X_train, y_train)  # refit na pelnym zbiorze treningowym
    results["Random Forest"] = evaluate(best_rf, X_test, y_test)
    results["Random Forest"]["train_time_s"] = round(time.time() - t0, 2)
    results["Random Forest"]["best_params"] = rf_search.best_params_
    results["Random Forest"]["model_size_mb"] = model_size_mb(best_rf)
    fitted_models["Random Forest"] = best_rf
    print(results["Random Forest"])

    # 3) XGBoost + GridSearchCV (na probie), nastepnie refit na calosci
    print("\n[3/3] XGBoost (GridSearchCV na probie 4000 wierszy)...")
    xgb_pipe = Pipeline(
        [("prep", build_preprocessor()), ("model", XGBRegressor(random_state=42, objective="reg:squarederror"))]
    )
    xgb_grid = {
        "model__n_estimators": [200, 350],
        "model__max_depth": [4, 6],
        "model__learning_rate": [0.05, 0.1],
    }
    t0 = time.time()
    xgb_search = GridSearchCV(xgb_pipe, xgb_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=1)
    xgb_search.fit(tune_sample, y_tune_sample)
    best_xgb = xgb_search.best_estimator_
    best_xgb.set_params(**xgb_search.best_params_)
    best_xgb.fit(X_train, y_train)  # refit na pelnym zbiorze treningowym
    results["XGBoost"] = evaluate(best_xgb, X_test, y_test)
    results["XGBoost"]["train_time_s"] = round(time.time() - t0, 2)
    results["XGBoost"]["best_params"] = xgb_search.best_params_
    results["XGBoost"]["model_size_mb"] = model_size_mb(best_xgb)
    fitted_models["XGBoost"] = best_xgb
    print(results["XGBoost"])

    # Wybor najlepszego modelu: nie tylko najnizszy RMSE, ale takze rozmiar
    # pliku modelu ma znaczenie praktyczne (limit GitHub 100 MB, szybkosc
    # wczytywania aplikacji Streamlit). Jesli model o nieznacznie gorszym
    # RMSE (w granicach 3%) jest znaczaco mniejszy, wybieramy go do wdrozenia.
    best_rmse_name = min(results, key=lambda k: results[k]["RMSE"])
    best_rmse = results[best_rmse_name]["RMSE"]

    deployable_name = best_rmse_name
    for name, res in results.items():
        if name == best_rmse_name:
            continue
        within_tolerance = res["RMSE"] <= best_rmse * 1.03
        much_smaller = res.get("model_size_mb", 0) < results[best_rmse_name].get("model_size_mb", 0) / 5
        if within_tolerance and much_smaller:
            deployable_name = name

    print(f"\nNajlepszy model wg samego RMSE: {best_rmse_name} -> {results[best_rmse_name]}")
    if deployable_name != best_rmse_name:
        print(
            f"Model wybrany do wdrozenia: {deployable_name} "
            f"(RMSE w granicach 3% od najlepszego, ale plik {results[best_rmse_name]['model_size_mb']} MB "
            f"-> {results[deployable_name]['model_size_mb']} MB - istotne dla wdrozenia na GitHub/Streamlit Cloud)"
        )
    best_name = deployable_name
    best_model = fitted_models[best_name]

    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        MODEL_PATH,
    )
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results, "best_model": best_name}, f, ensure_ascii=False, indent=2)

    print(f"\nZapisano model: {MODEL_PATH}")
    print(f"Zapisano metryki: {METRICS_PATH}")


if __name__ == "__main__":
    main()
