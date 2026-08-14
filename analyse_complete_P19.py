# -*- coding: utf-8 -*-
"""
=====================================================================================
 PROJET P19 - Prevision des prix des denrees de base sur les marches guineens
 Master 1 - Fouille de Donnees - Universite Kofi Annan de Guinee
=====================================================================================
Script unique regroupant l'integralite de l'analyse, conformement a la methodologie
CRISP-DM du cours :
    1. Generation / chargement des donnees
    2. Analyse exploratoire (EDA)
    3. Preparation (nettoyage, feature engineering)
    4. Modelisation (Holt-Winters, Random Forest, Gradient Boosting)
    5. Evaluation (RMSE, MAE, R2) et prevision a 12 semaines

Note methodologique : cet environnement de developpement n'a pas d'acces reseau.
  - Les donnees reelles HDX / PAM Food Prices n'ont donc pas pu etre telechargees ;
    elles sont generees de maniere synthetique en respectant les ordres de grandeur,
    la saisonnalite et les chocs macro-economiques reellement documentes pour la
    Guinee sur la periode 2019-2024 (methode explicitement autorisee par le cadre
    du cours lorsque la donnee locale n'est pas directement telechargeable).
  - statsmodels (SARIMA) et Prophet n'ont pas pu etre installes hors-ligne : un
    lissage exponentiel saisonnier (Holt-Winters) implemente manuellement joue le
    role de modele statistique de reference.
  - xgboost n'a pas pu etre installe hors-ligne : HistGradientBoostingRegressor
    (scikit-learn), meme famille d'algorithme (arbres de boosting sur histogrammes
    de gradients), est utilise en remplacement.

Auteur : Etudiant Master 1 - Fouille de Donnees - Sujet P19
=====================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RNG = np.random.default_rng(42)
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
COLORS = {"riz": "#2E86AB", "huile": "#E67E22", "sucre": "#27AE60"}

DATA_DIR = "data"
FIG_DIR = "figures"
os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# =====================================================================================
# 1. GENERATION / CHARGEMENT DES DONNEES
# =====================================================================================
def generate_data():
    """Genere le jeu de donnees synthetique hebdomadaire (5 marches x 3 denrees)."""
    start = pd.Timestamp("2019-01-06")
    end = pd.Timestamp("2024-12-30")
    dates = pd.date_range(start, end, freq="W-SUN")
    n = len(dates)

    markets = {
        "Conakry (Madina)": {"riz": 1.00, "huile": 1.00, "sucre": 1.00},
        "Kindia":            {"riz": 1.04, "huile": 1.05, "sucre": 1.03},
        "Labe":              {"riz": 1.10, "huile": 1.12, "sucre": 1.09},
        "Kankan":            {"riz": 1.08, "huile": 1.11, "sucre": 1.07},
        "Nzerekore":         {"riz": 1.13, "huile": 1.15, "sucre": 1.12},
    }
    base_price = {"riz": 4500.0, "huile": 9000.0, "sucre": 6500.0}
    annual_inflation = {"riz": 0.09, "huile": 0.12, "sucre": 0.08}
    unit = {"riz": "GNF/kg", "huile": "GNF/L", "sucre": "GNF/kg"}
    t_years = (dates - start).days / 365.25

    def seasonality(commodity, dates):
        month = dates.month
        doy = dates.dayofyear
        if commodity == "riz":
            s = -0.06 * np.cos(2 * np.pi * (doy - 300) / 365) + 0.02 * np.sin(2 * np.pi * (doy - 150) / 365)
        elif commodity == "huile":
            s = 0.03 * np.sin(2 * np.pi * (doy - 60) / 365)
        else:  # sucre
            s = 0.025 * np.sin(2 * np.pi * (doy - 100) / 365) + 0.02 * np.cos(2 * np.pi * (doy - 340) / 365)
        rainy = np.isin(month, [6, 7, 8, 9]).astype(float)
        return s + 0.02 * rainy

    def shocks(dates):
        s = np.zeros(len(dates))
        covid = (dates >= "2020-03-01") & (dates <= "2020-09-15")
        s += np.where(covid, 0.07, 0.0)
        ukraine = (dates >= "2022-02-20") & (dates <= "2023-01-31")
        ramp = np.clip((dates - pd.Timestamp("2022-02-20")).days / 60, 0, 1)
        s += np.where(ukraine, 0.10 * ramp, 0.0)
        fuel = (dates >= "2023-11-01") & (dates <= "2024-04-30")
        s += np.where(fuel, 0.05, 0.0)
        return s

    rows = []
    for commodity in ["riz", "huile", "sucre"]:
        trend = base_price[commodity] * (1 + annual_inflation[commodity]) ** t_years
        season = seasonality(commodity, dates)
        shock = shocks(dates)
        common_ar = np.zeros(n)
        for i in range(1, n):
            common_ar[i] = 0.6 * common_ar[i - 1] + RNG.normal(0, 0.012)
        for mkt, info in markets.items():
            mult = info[commodity]
            idio_noise = RNG.normal(0, 0.018, size=n)
            price = trend * mult * (1 + season + shock + common_ar + idio_noise)
            price = np.maximum(price, base_price[commodity] * 0.3)
            rows.append(pd.DataFrame({
                "date": dates, "marche": mkt, "denree": commodity,
                "unite": unit[commodity], "prix_gnf": price.round(0),
            }))

    data = pd.concat(rows, ignore_index=True)
    mask = RNG.random(len(data)) < 0.035
    data.loc[mask, "prix_gnf"] = np.nan
    dupes = data.sample(frac=0.01, random_state=1)
    data = pd.concat([data, dupes], ignore_index=True)
    data = data.sort_values(["denree", "marche", "date"]).reset_index(drop=True)
    data.to_csv(os.path.join(DATA_DIR, "raw", "guinee_prix_denrees_raw.csv"), index=False)
    return data


# =====================================================================================
# 2. ANALYSE EXPLORATOIRE (EDA)
# =====================================================================================
def run_eda(df):
    print("=== Dimensions ===", df.shape)
    print("=== Valeurs manquantes ===\n", df.isna().sum())
    desc = df.groupby("denree")["prix_gnf"].describe()
    print("=== Statistiques descriptives ===\n", desc)
    desc.to_csv(os.path.join(DATA_DIR, "stats_descriptives.csv"))

    df = df.drop_duplicates(subset=["date", "marche", "denree"])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, com in zip(axes, ["riz", "huile", "sucre"]):
        vals = df.loc[df.denree == com, "prix_gnf"].dropna()
        ax.hist(vals, bins=35, color=COLORS[com], alpha=0.85)
        ax.set_title(f"{com.capitalize()} ({df.loc[df.denree==com,'unite'].iloc[0]})")
        ax.set_xlabel("Prix (GNF)"); ax.set_ylabel("Frequence")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "distributions.png")); plt.close()

    national = df.groupby(["denree", "date"])["prix_gnf"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for com in ["riz", "huile", "sucre"]:
        sub = national[national.denree == com].sort_values("date")
        ax.plot(sub.date, sub.prix_gnf, label=com.capitalize(), color=COLORS[com], linewidth=1.4)
    ax.set_ylabel("Prix moyen national (GNF)"); ax.legend()
    ax.set_title("Evolution des prix moyens nationaux (2019-2024)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "series_nationales.png")); plt.close()

    pivot = national.pivot(index="date", columns="denree", values="prix_gnf").interpolate()
    corr = pivot.corr()
    corr.to_csv(os.path.join(DATA_DIR, "matrice_correlation.csv"))
    print("=== Matrice de correlation ===\n", corr)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels([c.capitalize() for c in corr.columns])
    ax.set_yticklabels([c.capitalize() for c in corr.columns])
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                     color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    plt.colorbar(im, fraction=0.046, pad=0.04); ax.set_title("Correlation entre prix des denrees")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "correlation.png")); plt.close()

    miss = df.groupby(["denree", "marche"]).apply(lambda g: g["prix_gnf"].isna().mean() * 100)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    miss.unstack().plot(kind="bar", ax=ax)
    ax.set_ylabel("% valeurs manquantes"); ax.legend(title="Marche", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "valeurs_manquantes.png")); plt.close()

    df.to_csv(os.path.join(DATA_DIR, "processed", "guinee_prix_denrees_clean.csv"), index=False)
    return df, national


# =====================================================================================
# 3. PREPARATION : FEATURE ENGINEERING
# =====================================================================================
def make_features(s):
    d = pd.DataFrame({"y": s})
    for lag in [1, 2, 4, 8, 12, 52]:
        d[f"lag_{lag}"] = d["y"].shift(lag)
    d["roll_mean_4"] = d["y"].shift(1).rolling(4).mean()
    d["roll_mean_12"] = d["y"].shift(1).rolling(12).mean()
    d["roll_std_4"] = d["y"].shift(1).rolling(4).std()
    d["month"] = d.index.month
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["is_rainy"] = d["month"].isin([6, 7, 8, 9]).astype(int)
    d["trend_idx"] = np.arange(len(d))
    return d.dropna()


# =====================================================================================
# 4. MODELISATION : Holt-Winters / Random Forest / Gradient Boosting
# =====================================================================================
def holt_winters_fit_forecast(y_train, season_len, h, alpha=0.25, beta=0.05, gamma=0.3):
    """Lissage exponentiel saisonnier additif (implementation manuelle, statsmodels
    indisponible hors-ligne) - joue le role de modele statistique de reference (SARIMA)."""
    y = y_train.values.astype(float)
    n = len(y)
    level = np.mean(y[:season_len])
    trend = (np.mean(y[season_len:2 * season_len]) - np.mean(y[:season_len])) / season_len
    seasonals = [y[i] - level for i in range(season_len)]
    for t in range(n):
        seas = seasonals[t] if t < season_len else seasonals[t - season_len]
        prev_level = level
        level = alpha * (y[t] - seas) + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        if t >= season_len:
            new_seas = gamma * (y[t] - level) + (1 - gamma) * seas
            seasonals.append(new_seas)
    forecasts = []
    for k in range(1, h + 1):
        s_idx = (n + k - 1) % season_len
        seas = seasonals[-season_len + s_idx] if len(seasonals) >= season_len else 0
        forecasts.append(level + k * trend + seas)
    return np.array(forecasts)


def run_modeling(national):
    series = {}
    for com in ["riz", "huile", "sucre"]:
        s = national[national.denree == com].set_index("date")["prix_gnf"].asfreq("W-SUN")
        series[com] = s.interpolate(method="linear").ffill().bfill()

    results = []
    forecasts_store, feature_importances = {}, {}

    for com in ["riz", "huile", "sucre"]:
        s = series[com]
        feat = make_features(s)
        split = int(len(feat) * 0.8)
        train, test = feat.iloc[:split], feat.iloc[split:]
        X_cols = [c for c in feat.columns if c != "y"]
        X_train, y_train = train[X_cols], train["y"]
        X_test, y_test = test[X_cols], test["y"]
        h = len(test)

        # -- Holt-Winters --
        hw_pred = pd.Series(
            holt_winters_fit_forecast(s.loc[:train.index[-1]], season_len=52, h=h), index=test.index
        )

        # -- Cible = variation hebdomadaire (evite le piege d'extrapolation des arbres) --
        delta_train = y_train.diff().dropna()
        X_train_d = X_train.loc[delta_train.index]
        lag1_test = feat.loc[test.index, "lag_1"]

        rf = RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=2, random_state=42)
        rf.fit(X_train_d, delta_train)
        rf_pred = pd.Series(lag1_test.values + rf.predict(X_test), index=test.index)

        gb = HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.06, random_state=42)
        gb.fit(X_train_d, delta_train)
        gb_pred = pd.Series(lag1_test.values + gb.predict(X_test), index=test.index)

        for name, pred in [("Holt-Winters (saisonnier)", hw_pred),
                            ("Random Forest", rf_pred),
                            ("Gradient Boosting (type XGBoost)", gb_pred)]:
            results.append({
                "denree": com, "modele": name,
                "RMSE": mean_squared_error(y_test, pred) ** 0.5,
                "MAE": mean_absolute_error(y_test, pred),
                "R2": r2_score(y_test, pred),
            })

        forecasts_store[com] = {"test_index": test.index, "y_test": y_test, "hw": hw_pred, "rf": rf_pred, "gb": gb_pred}
        # HistGradientBoostingRegressor n'expose pas feature_importances_ ; on utilise
        # celles de la Random Forest, comparable, pour l'analyse des variables explicatives.
        feature_importances[com] = pd.Series(rf.feature_importances_, index=X_cols).sort_values(ascending=False)

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(DATA_DIR, "resultats_modeles.csv"), index=False)
    print("=== Resultats des modeles ===\n", res_df.round(3))

    best = res_df.loc[res_df.groupby("denree")["RMSE"].idxmin()]
    best.to_csv(os.path.join(DATA_DIR, "meilleurs_modeles.csv"), index=False)
    print("=== Meilleur modele par denree ===\n", best)

    # Graphique previsions vs reel
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for ax, com in zip(axes, ["riz", "huile", "sucre"]):
        fc = forecasts_store[com]
        ax.plot(fc["test_index"], fc["y_test"], color="black", label="Reel (test)", linewidth=1.6)
        ax.plot(fc["test_index"], fc["hw"], "--", color="#8E44AD", label="Holt-Winters")
        ax.plot(fc["test_index"], fc["rf"], "--", color="#2E86AB", label="Random Forest")
        ax.plot(fc["test_index"], fc["gb"], "--", color="#E67E22", label="Gradient Boosting")
        ax.set_title(f"{com.capitalize()} - Prevision vs realise"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "previsions_vs_reel.png")); plt.close()

    # Graphique importance des variables
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, com in zip(axes, ["riz", "huile", "sucre"]):
        imp = feature_importances[com].head(8)[::-1]
        ax.barh(imp.index, imp.values, color=COLORS[com])
        ax.set_title(f"Variables importantes - {com.capitalize()}")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "importance_variables.png")); plt.close()

    # Prevision future a 12 semaines avec le meilleur modele par denree
    future_h = 12
    future_results = {}
    for com in ["riz", "huile", "sucre"]:
        s = series[com]
        feat_full = make_features(s)
        X_cols = [c for c in feat_full.columns if c != "y"]
        best_model_name = best.loc[best.denree == com, "modele"].values[0]
        future_dates = pd.date_range(s.index[-1] + pd.Timedelta(weeks=1), periods=future_h, freq="W-SUN")

        if "Holt-Winters" in best_model_name:
            fc = holt_winters_fit_forecast(s, season_len=52, h=future_h)
        else:
            model = (RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=2, random_state=42)
                     if "Random Forest" in best_model_name
                     else HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.06, random_state=42))
            delta_full = feat_full["y"].diff().dropna()
            model.fit(feat_full.loc[delta_full.index, X_cols], delta_full)
            hist = s.copy()
            preds = []
            for _ in range(future_h):
                tmp = make_features(hist)
                x_last = tmp[X_cols].iloc[[-1]]
                p = hist.iloc[-1] + model.predict(x_last)[0]
                preds.append(p)
                hist.loc[hist.index[-1] + pd.Timedelta(weeks=1)] = p
            fc = np.array(preds)
        future_results[com] = pd.Series(fc, index=future_dates)

    fut_df = pd.concat(future_results, axis=1)
    fut_df.to_csv(os.path.join(DATA_DIR, "prevision_12_semaines.csv"))
    print("=== Prevision 12 semaines ===\n", fut_df.round(0))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    for ax, com in zip(axes, ["riz", "huile", "sucre"]):
        s = series[com]
        ax.plot(s.index[-30:], s.iloc[-30:], color="black", label="Historique recent")
        ax.plot(future_results[com].index, future_results[com].values, "--", color=COLORS[com], marker="o", markersize=3, label="Prevision (12 semaines)")
        ax.axvline(s.index[-1], color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"{com.capitalize()} - Prevision a 12 semaines"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "prevision_future.png")); plt.close()

    return res_df, fut_df


# =====================================================================================
# MAIN
# =====================================================================================
if __name__ == "__main__":
    print(">> Etape 1/4 : generation des donnees")
    raw = generate_data()

    print("\n>> Etape 2/4 : analyse exploratoire")
    clean, national = run_eda(raw)

    print("\n>> Etape 3-4/4 : preparation, modelisation et evaluation")
    resultats, previsions = run_modeling(national)

    print("\n>> Analyse complete terminee. Fichiers et figures dans data/ et figures/.")
