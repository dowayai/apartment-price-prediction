"""
eda.py
------
Analiza eksploracyjna danych (EDA) - generuje wykresy do rozdziału 2 pracy
(przygotowanie danych i analiza rynku).

Wyjście: pliki PNG w folderze figures/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("data/cleaned_apartments.csv")

# 1. Rozkład cen mieszkań
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=60, kde=True, color="#2E86AB")
plt.title("Rozkład cen mieszkań")
plt.xlabel("Cena (PLN)")
plt.ylabel("Liczba ofert")
plt.tight_layout()
plt.savefig("figures/01_price_distribution.png", dpi=150)
plt.close()

# 2. Cena za m2 wg miasta
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="city", y="price_per_m", palette="Set2", hue="city", legend=False)
plt.title("Cena za m² według miasta")
plt.xlabel("Miasto")
plt.ylabel("Cena za m² (PLN)")
plt.tight_layout()
plt.savefig("figures/02_price_per_m_by_city.png", dpi=150)
plt.close()

# 3. Zależność ceny od metrażu
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df.sample(3000, random_state=42), x="sq", y="price", hue="city", alpha=0.5, s=20)
plt.title("Cena mieszkania a metraż")
plt.xlabel("Powierzchnia (m²)")
plt.ylabel("Cena (PLN)")
plt.tight_layout()
plt.savefig("figures/03_price_vs_sq.png", dpi=150)
plt.close()

# 4. Cena a odległość od centrum
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df.sample(3000, random_state=42),
    x="distance_to_center_km", y="price_per_m", hue="city", alpha=0.5, s=20
)
plt.title("Cena za m² a odległość od centrum miasta")
plt.xlabel("Odległość od centrum (km)")
plt.ylabel("Cena za m² (PLN)")
plt.tight_layout()
plt.savefig("figures/04_price_vs_distance.png", dpi=150)
plt.close()

# 5. Macierz korelacji cech numerycznych
plt.figure(figsize=(7, 6))
num_cols = ["rooms", "floor", "sq", "building_age", "distance_to_center_km", "price"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Macierz korelacji cech numerycznych")
plt.tight_layout()
plt.savefig("figures/05_correlation_heatmap.png", dpi=150)
plt.close()

print("Zapisano 5 wykresów EDA w folderze figures/")
print("\nPodstawowe statystyki:")
print(df.describe(include="all").T)
