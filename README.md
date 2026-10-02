# Prognozowanie cen mieszkań w Polsce (ML + SHAP + Streamlit)

> **EN summary:** End-to-end machine learning project predicting secondary-market apartment prices in Warsaw, Kraków and Poznań (~21k listings). Linear Regression, Random Forest and XGBoost are compared (MAE, RMSE, R²; best R² ≈ 0.87). Predictions are explained with SHAP and served in an interactive Streamlit web app. Built as an engineering thesis (Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit).

Projekt inżynierski (Społeczna Akademia Nauk): system predykcji cen mieszkań na polskim rynku wtórnym z wyjaśnialnością decyzji modelu (XAI) oraz interaktywną aplikacją webową.

**Demo online:** _link do Streamlit Cloud (do uzupełnienia po wdrożeniu)_

## Co robi aplikacja

- szacuje cenę mieszkania na podstawie parametrów (miasto, metraż, liczba pokoi, piętro, wiek budynku, odległość od centrum),
- porównuje cenę ofertową z wyceną modelu (**„Okazja czy przepłacanie?”**),
- pokazuje wykres **SHAP waterfall**: o ile złotych każda cecha podnosi lub obniża wycenę.

## Dane

Zbiór [House Prices in Poland](https://www.kaggle.com/datasets/dawidcegielski/house-prices-in-poland) (Kaggle, autor: dawidcegielski) w wersji wzbogaconej o dane geograficzne (dzielnice, odległość od centrum) z repozytorium [am-tropin/poland-apartment-prices](https://github.com/am-tropin/poland-apartment-prices).
Ok. 21,6 tys. ogłoszeń sprzedaży (Warszawa, Kraków, Poznań).

Czyszczenie danych (`data_prep.py`): usunięcie rekordów z nierealnym rokiem budowy (< 1800) oraz skrajnych wartości (percentyle 0,5–99,5%) dla ceny, metrażu i ceny za m².

## Metodyka

- Cechy: `rooms`, `floor`, `sq`, `building_age`, `distance_to_center_km`, `city`.
- Podział train/test 80/20 (`random_state=42`); cena za m² **nie** jest używana jako cecha (brak wycieku danych).
- Strojenie hiperparametrów: `GridSearchCV` na próbie 4000 wierszy, następnie trenowanie na pełnym zbiorze uczącym.

## Wyniki (zbiór testowy)

| Model             | MAE (PLN) | RMSE (PLN) | R²    | Rozmiar modelu |
|-------------------|-----------|------------|-------|----------------|
| Linear Regression | 113 702   | 180 129    | 0.736 | < 0.1 MB |
| Random Forest     | **67 862**| **125 265**| **0.872** | 65.6 MB |
| XGBoost           | 72 148    | 125 366    | 0.872 | **1.5 MB** |

Random Forest wypada minimalnie lepiej pod względem MAE, ale XGBoost daje praktycznie tę samą dokładność (różnica RMSE < 0,1%) przy ~45× mniejszym modelu. Do wdrożenia wybrano więc **XGBoost** (limity rozmiaru plików na GitHub i czas ładowania aplikacji). Pełne wyniki i hiperparametry: `models/metrics.json`.

## Wyjaśnialność (SHAP)

Wykresy w katalogu `figures/`: ważność cech, wykres podsumowujący SHAP i przykład lokalnego wyjaśnienia dla pojedynczego mieszkania.

## Ograniczenia

- Ceny pochodzą z **ogłoszeń** (ceny ofertowe), a nie z transakcji.
- Dane obejmują tylko 3 miasta; model nie uwzględnia m.in. stanu technicznego, standardu wykończenia ani dokładnej lokalizacji (dzielnice nie są cechą modelu).
- Możliwy rozwój: cechy geolokalizacyjne, automatyczne pobieranie bieżących ofert.

## Uruchomienie lokalne

```bash
pip install -r requirements.txt

python data_prep.py        # 1. czyszczenie danych
python eda.py              # 2. analiza eksploracyjna (wykresy)
python train_model.py      # 3. trenowanie i porównanie modeli
python shap_analysis.py    # 4. analiza SHAP (wykresy)

streamlit run app.py       # 5. aplikacja webowa
```

## Struktura projektu

```
.
├── app.py               # aplikacja Streamlit
├── data_prep.py         # czyszczenie danych i feature engineering
├── eda.py               # analiza eksploracyjna
├── train_model.py       # trenowanie i porównanie modeli
├── shap_analysis.py     # analiza SHAP
├── requirements.txt
├── data/                # dane surowe i oczyszczone
├── models/              # best_model.pkl, metrics.json
└── figures/             # wykresy EDA i SHAP
```

## Technologie

Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · Streamlit · matplotlib · seaborn
