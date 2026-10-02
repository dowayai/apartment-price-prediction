"""
app.py
------
Etap 5: Aplikacja webowa (Streamlit).

Prosty interfejs webowy do szacowania ceny mieszkania w Polsce:
- uzytkownik wprowadza parametry mieszkania w panelu bocznym,
- aplikacja natychmiast zwraca szacowana cene z wytrenowanego modelu,
- wyswietlany jest wykres SHAP waterfall, ktory wyjasnia, jak kazda cecha
  wplynela na te konkretna wycene (lokalna interpretowalnosc XAI),
- dodatkowo: prosty "radar inwestycyjny" - porownanie ceny ofertowej
  z szacowana wartoscia rynkowa.

Uruchomienie lokalne:  streamlit run app.py
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Wycena mieszkań PL", page_icon="🏠", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")


@st.cache_data
def load_reference_data():
    return pd.read_csv("data/cleaned_apartments.csv")


bundle = load_model()
pipeline = bundle["model"]
model_name = bundle["model_name"]
numeric_features = bundle["numeric_features"]
categorical_features = bundle["categorical_features"]
prep = pipeline.named_steps["prep"]
model = pipeline.named_steps["model"]
explainer = shap.TreeExplainer(model)

df_ref = load_reference_data()

REFERENCE_YEAR = 2024  # musi byc zgodne z data_prep.py

st.title("🏠 Wycena mieszkań w Polsce")
st.caption(f"Model uczenia maszynowego: **{model_name}** · dane: Warszawa, Kraków, Poznań")

with st.sidebar:
    st.header("Parametry mieszkania")
    city = st.selectbox("Miasto", sorted(df_ref["city"].unique()))
    rooms = st.slider("Liczba pokoi", 1, 8, 3)
    floor = st.slider("Piętro", 0, 15, 2)
    sq = st.slider("Powierzchnia (m²)", 15, 250, 55)
    year = st.slider("Rok budowy", 1900, 2024, 2015)
    distance = st.slider("Odległość od centrum (km)", 0.0, 25.0, 5.0, step=0.1)

building_age = REFERENCE_YEAR - year

input_df = pd.DataFrame(
    [
        {
            "rooms": rooms,
            "floor": floor,
            "sq": sq,
            "building_age": building_age,
            "distance_to_center_km": distance,
            "city": city,
        }
    ]
)

X_input_transformed = prep.transform(input_df)
predicted_price = float(model.predict(X_input_transformed)[0])

col1, col2 = st.columns(2)
col1.metric("Szacowana cena", f"{predicted_price:,.0f} PLN")
col2.metric("Cena za m²", f"{predicted_price / sq:,.0f} PLN/m²")

st.divider()

# --- "Radar inwestycyjny": Okazja czy przepłacanie? ---
st.subheader("📊 Okazja czy przepłacanie?")
asking_price = st.number_input(
    "Podaj cenę ofertową konkretnego ogłoszenia (opcjonalnie), aby porównać z wyceną modelu",
    min_value=0,
    value=0,
    step=10_000,
)
if asking_price > 0:
    diff_pct = (predicted_price - asking_price) / predicted_price * 100
    if diff_pct > 5:
        st.success(
            f"💚 Super okazja! Cena ofertowa jest o **{diff_pct:.1f}%** niższa "
            f"niż szacowana wartość rynkowa ({predicted_price:,.0f} PLN)."
        )
    elif diff_pct < -5:
        st.error(
            f"🔴 Możliwe przepłacenie: cena ofertowa jest o **{abs(diff_pct):.1f}%** wyższa "
            f"niż szacowana wartość rynkowa ({predicted_price:,.0f} PLN)."
        )
    else:
        st.info("🟡 Cena ofertowa jest zgodna z szacowaną wartością rynkową (±5%).")

st.divider()

# --- Lokalne wyjaśnienie SHAP ---
st.subheader("🔍 Dlaczego taka cena?")
st.caption(
    "Wykres SHAP waterfall pokazuje, o ile złotych każda cecha zwiększa lub "
    "zmniejsza cenę względem wartości bazowej modelu (średniej ceny w zbiorze treningowym)."
)

feature_names = prep.get_feature_names_out()
feature_names = [f.replace("num__", "").replace("cat__city_", "city=") for f in feature_names]

shap_values = explainer(X_input_transformed)
shap_values.feature_names = feature_names

fig = plt.figure()
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig, bbox_inches="tight")
plt.close(fig)

st.divider()
st.caption(
    "Praca inżynierska: *Prognozowanie cen mieszkań z wykorzystaniem metod uczenia maszynowego*. "
    "Dane źródłowe: Kaggle — House Prices in Poland (dawidcegielski), "
    "przetworzone przez repozytorium am-tropin/poland-apartment-prices."
)
