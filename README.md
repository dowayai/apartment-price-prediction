# Prognozowanie cen mieszkań z wykorzystaniem metod uczenia maszynowego

Praca inżynierska — system predykcji cen mieszkań na polskim rynku wtórnym
(Warszawa, Kraków, Poznań) z wyjaśnialnością decyzji modelu (XAI/SHAP)
oraz interaktywną aplikacją webową (Streamlit).

## Struktura projektu

```
.
├── app.py                 # aplikacja webowa Streamlit
├── data_prep.py            # czyszczenie danych i feature engineering
├── eda.py                  # analiza eksploracyjna (wykresy do rozdz. 2)
├── train_model.py          # trenowanie i porównanie modeli
├── shap_analysis.py        # analiza wyjaśnialności SHAP (rozdz. 3)
├── requirements.txt
├── data/
│   ├── poland_apartments_completed.csv   # dane źródłowe (surowe)
│   └── cleaned_apartments.csv            # dane po czyszczeniu
├── models/
│   ├── best_model.pkl       # zapisany najlepszy model (pipeline)
│   └── metrics.json         # wyniki porównania modeli
└── figures/                 # wykresy EDA + SHAP do pracy dyplomowej
```

## Źródło danych

Dane pochodzą ze zbioru [House Prices in Poland](https://www.kaggle.com/datasets/dawidcegielski/house-prices-in-poland)
(Kaggle, autor: dawidcegielski), w wersji wzbogaconej o dane geograficzne
(dzielnice, odległość od centrum miasta) opracowanej w repozytorium
[am-tropin/poland-apartment-prices](https://github.com/am-tropin/poland-apartment-prices).
Zbiór obejmuje ok. 21,6 tys. ogłoszeń sprzedaży mieszkań w Warszawie,
Krakowie i Poznaniu.

**Ważne:** przy pisaniu rozdziału o danych źródłowych należy zacytować
oryginalny zbiór Kaggle oraz repozytorium, z którego pochodzi wersja
przetworzona.

## Uruchomienie lokalne

```bash
pip install -r requirements.txt

python data_prep.py        # 1. czyszczenie danych
python eda.py               # 2. analiza eksploracyjna (wykresy)
python train_model.py       # 3. trenowanie i porównanie modeli
python shap_analysis.py     # 4. analiza SHAP (wykresy)

streamlit run app.py        # 5. uruchomienie aplikacji webowej
```

## Wyniki porównania modeli

| Model              | MAE (PLN) | RMSE (PLN) | R²    | Rozmiar pliku modelu |
|---------------------|-----------|------------|-------|----------------------|
| Linear Regression    | 113 702   | 180 129    | 0.736 | < 0.1 MB |
| Random Forest        | **67 862**| **125 265**| **0.872** | 65.6 MB |
| XGBoost              | 72 148    | 125 366    | 0.872 | **1.5 MB** |

Random Forest osiągnął nieznacznie lepszy wynik MAE/RMSE, jednak jego
wytrenowany model waży **65,6 MB** — wobec **1,5 MB** dla XGBoost, przy
praktycznie identycznej dokładności (różnica RMSE < 0,1%). Ze względu na
ograniczenia wdrożeniowe (limit rozmiaru pliku na GitHub, czas wczytywania
aplikacji Streamlit Cloud) do wdrożenia w aplikacji webowej wybrano
**XGBoost**. To świadoma decyzja inżynierska warta opisania w rozdziale
z wnioskami — pokazuje kompromis między dokładnością a praktycznością
wdrożenia modelu. Pełne wyniki i dobrane hiperparametry: `models/metrics.json`.

## Wdrożenie (deployment)

Aplikację można bezpłatnie opublikować w internecie za pomocą
[Streamlit Community Cloud](https://streamlit.io/cloud), łącząc się
bezpośrednio z tym repozytorium GitHub — patrz instrukcja krok po kroku
przekazana osobno.
