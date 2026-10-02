"""
data_prep.py
------------
Etap 1: Zbieranie i przygotowanie danych (Data Engineering)

Źródło danych: dataset "House Prices in Poland" (Kaggle, autor: dawidcegielski),
w wersji przetworzonej i wzbogaconej o dane geograficzne (dzielnice, odległość
od centrum miasta) przez repozytorium am-tropin/poland-apartment-prices (GitHub).
Link do oryginalnego zbioru: https://www.kaggle.com/datasets/dawidcegielski/house-prices-in-poland

Skrypt:
1. Wczytuje surowe dane (ceny mieszkań w Warszawie, Krakowie i Poznaniu).
2. Naprawia błąd kodowania znaków w nazwie miasta "Poznań".
3. Usuwa rekordy będące oczywistymi błędami wprowadzania danych (np. rok
   budowy sprzed XIX wieku, ekstremalne wartości ceny za m2).
4. Tworzy nowe cechy (feature engineering): wiek budynku, cenę za m2.
5. Zapisuje oczyszczony zbiór do data/cleaned_apartments.csv gotowy do EDA
   i trenowania modeli.
"""

import pandas as pd

RAW_PATH = "data/poland_apartments_completed.csv"
CLEAN_PATH = "data/cleaned_apartments.csv"

# rok, do którego odnosimy wiek budynku (rok zebrania danych)
REFERENCE_YEAR = 2024


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)

    # 1) błąd kodowania: "Poznañ" -> "Poznań"
    df["city"] = df["city"].replace({"Poznañ": "Poznań"})

    # 2) usuwamy rekordy z nierealnym rokiem budowy (błędy wpisu, np. rok 1390)
    df = df[df["year"] >= 1800]

    # 3) usuwamy skrajne wartości (0.5%-99.5% percentyla) dla ceny, metrażu
    #    i ceny za m2 - eliminujemy błędy wpisu (np. 8 m2 za 1.22 mln zł)
    #    oraz nietypowe oferty luksusowe, które zaburzałyby model
    for col in ["price", "sq", "price_per_m"]:
        lo, hi = df[col].quantile([0.005, 0.995])
        df = df[(df[col] >= lo) & (df[col] <= hi)]

    df = df.reset_index(drop=True)

    n_after = len(df)
    print(
        f"Czyszczenie danych: {n_before} -> {n_after} rekordów "
        f"(usunięto {n_before - n_after}, {(n_before - n_after) / n_before:.2%})"
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # wiek budynku w momencie zebrania danych
    df["building_age"] = REFERENCE_YEAR - df["year"]
    # odległość od centrum miasta (już policzona w danych źródłowych jako "radius")
    df = df.rename(columns={"radius": "distance_to_center_km"})
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "city",
        "district",
        "rooms",
        "floor",
        "sq",
        "year",
        "building_age",
        "distance_to_center_km",
        "price_per_m",
        "price",
    ]
    return df[keep]


def main():
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)
    df = select_columns(df)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Zapisano oczyszczony zbiór: {CLEAN_PATH} ({df.shape[0]} wierszy, {df.shape[1]} kolumn)")
    print(df.head())


if __name__ == "__main__":
    main()
