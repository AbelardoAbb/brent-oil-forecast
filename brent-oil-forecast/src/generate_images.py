import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

DARK = '#0d1117'
ACCENT = '#f0a500'
GREEN = '#2ea44f'
BLUE = '#58a6ff'
RED = '#f85149'
GRAY = '#8b949e'
WHITE = '#e6edf3'
CARD = '#161b22'

plt.rcParams.update({
    'figure.facecolor': DARK,
    'axes.facecolor': CARD,
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': WHITE,
    'xtick.color': GRAY,
    'ytick.color': GRAY,
    'text.color': WHITE,
    'grid.color': '#21262d',
    'grid.linewidth': 0.6,
    'font.family': 'monospace',
    'legend.facecolor': CARD,
    'legend.edgecolor': '#30363d',
    'legend.labelcolor': WHITE,
})

IMG = '/home/claude/brent-oil-forecast/images'

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
df_raw = pd.read_csv('/home/claude/brent-oil-forecast/data/BrentOilPrices.csv')
df_raw['Date'] = pd.to_datetime(df_raw['Date'], dayfirst=False)
df_raw = df_raw.sort_values('Date').set_index('Date')
df = df_raw.copy()

# ── 01 SERIE COMPLETA ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(DARK)
ax.plot(df.index, df['Price'], color=ACCENT, linewidth=0.9, alpha=0.95)
ax.fill_between(df.index, df['Price'], alpha=0.12, color=ACCENT)
ax.set_title('Brent Crude Oil Price  (1987 – 2022)', fontsize=16, color=WHITE, pad=14, fontweight='bold')
ax.set_ylabel('Price (USD/barrel)', color=GRAY, fontsize=11)
ax.set_xlabel('')

events = [
    ('1990-08-02', 'Gulf War'),
    ('2008-07-03', 'Peak $143'),
    ('2009-01-15', 'Financial Crisis'),
    ('2014-11-01', 'OPEC Shock'),
    ('2020-04-21', 'COVID-19'),
]
for d, label in events:
    xd = pd.to_datetime(d)
    yval = df.loc[df.index.asof(xd), 'Price'] if xd in df.index else df['Price'].iloc[0]
    ax.axvline(xd, color=RED, linewidth=0.8, linestyle='--', alpha=0.6)
    ax.text(xd, df['Price'].max()*0.92, label, rotation=90, fontsize=7.5,
            color=RED, alpha=0.85, va='top', ha='right')

ax.yaxis.grid(True)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f'{IMG}/01_serie_completa.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("✔ 01_serie_completa.png")

# ── 02 DECOMPOSIÇÃO ────────────────────────────────────────────────────────────
df_m = df['Price'].resample('ME').mean().dropna()
decomp = sm.tsa.seasonal_decompose(df_m, model='additive', period=12)

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.patch.set_facecolor(DARK)
fig.suptitle('Seasonal Decomposition  (Monthly, Additive)', fontsize=15, color=WHITE, fontweight='bold', y=1.01)

labels = ['Observed', 'Trend', 'Seasonal', 'Residual']
colors = [ACCENT, BLUE, GREEN, RED]
data_parts = [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid]

for ax, data, label, c in zip(axes, data_parts, labels, colors):
    ax.plot(data.index, data, color=c, linewidth=1.1)
    ax.set_ylabel(label, color=GRAY, fontsize=10)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f'{IMG}/02_decomposicao.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("✔ 02_decomposicao.png")

# ── 03 TESTE ADF ───────────────────────────────────────────────────────────────
adf_result = adfuller(df['Price'], autolag='AIC')
stat, pval = adf_result[0], adf_result[1]
crits = adf_result[4]

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.patch.set_facecolor(DARK)
fig.suptitle('Stationarity — Augmented Dickey-Fuller Test', fontsize=14, color=WHITE, fontweight='bold')

# rolling stats
roll = df['Price'].rolling(365)
axes[0].plot(df.index, df['Price'], color=ACCENT, linewidth=0.7, alpha=0.7, label='Price')
axes[0].plot(roll.mean().index, roll.mean(), color=BLUE, linewidth=1.5, label='Rolling Mean (1Y)')
axes[0].plot(roll.std().index, roll.std(), color=RED, linewidth=1.5, label='Rolling Std (1Y)')
axes[0].legend(fontsize=9)
axes[0].set_title('Rolling Statistics', color=GRAY, fontsize=11)
axes[0].yaxis.grid(True); axes[0].set_axisbelow(True)

# ADF summary bar
levels = ['1%', '5%', '10%', 'ADF Stat']
vals   = [crits['1%'], crits['5%'], crits['10%'], stat]
bar_c  = [GRAY, GRAY, GRAY, GREEN if pval < 0.05 else RED]
bars   = axes[1].barh(levels, vals, color=bar_c, alpha=0.85, height=0.5)
axes[1].set_title(f'ADF Statistic vs Critical Values\np-value = {pval:.4f}  →  {"Stationary ✔" if pval < 0.05 else "Non-Stationary ✘"}',
                  color=WHITE, fontsize=11)
axes[1].axvline(0, color=WHITE, linewidth=0.5)
axes[1].xaxis.grid(True); axes[1].set_axisbelow(True)
for bar, v in zip(bars, vals):
    axes[1].text(v + 0.3, bar.get_y() + bar.get_height()/2, f'{v:.2f}',
                 va='center', fontsize=9, color=WHITE)

plt.tight_layout()
plt.savefig(f'{IMG}/03_adf_test.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("✔ 03_adf_test.png")

# ── FEATURE ENGINEERING ────────────────────────────────────────────────────────
df_feat = df.copy()
df_feat['month']       = df_feat.index.month
df_feat['quarter']     = df_feat.index.quarter
df_feat['year']        = df_feat.index.year
df_feat['day_of_week'] = df_feat.index.dayofweek
df_feat['day_of_year'] = df_feat.index.dayofyear

for lag in [1, 2, 3, 5, 10, 20, 60, 120, 252]:
    df_feat[f'lag_{lag}'] = df_feat['Price'].shift(lag)

df_feat['ma_5']   = df_feat['Price'].shift(1).rolling(5).mean()
df_feat['ma_20']  = df_feat['Price'].shift(1).rolling(20).mean()
df_feat['ma_60']  = df_feat['Price'].shift(1).rolling(60).mean()
df_feat['ma_252'] = df_feat['Price'].shift(1).rolling(252).mean()
df_feat['std_20'] = df_feat['Price'].shift(1).rolling(20).std()
df_feat['std_60'] = df_feat['Price'].shift(1).rolling(60).std()
df_feat['ret_1']  = df_feat['Price'].pct_change(1)
df_feat['ret_5']  = df_feat['Price'].pct_change(5)

df_feat = df_feat.dropna()

steps = 252  # ~1 trading year
train = df_feat.iloc[:-steps]
test  = df_feat.iloc[-steps:]

feature_cols = [c for c in df_feat.columns if c != 'Price']
X_train, y_train = train[feature_cols], train['Price']
X_test,  y_test  = test[feature_cols],  test['Price']

# ── 04 TRAIN/TEST SPLIT ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
fig.patch.set_facecolor(DARK)
ax.plot(train.index, train['Price'], color=BLUE, linewidth=0.9, label=f'Train  ({len(train):,} days)')
ax.plot(test.index,  test['Price'],  color=ACCENT, linewidth=1.3, label=f'Test  ({len(test):,} days — last trading year)')
ax.axvline(test.index[0], color=GRAY, linewidth=1.2, linestyle='--', alpha=0.7)
ax.set_title('Train / Test Split  (Walk-Forward — no data leakage)', fontsize=14, color=WHITE, fontweight='bold', pad=12)
ax.set_ylabel('Price (USD/barrel)', color=GRAY, fontsize=10)
ax.legend(fontsize=10)
ax.yaxis.grid(True); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f'{IMG}/04_train_test_split.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("✔ 04_train_test_split.png")

# ── TRAIN MODELS ───────────────────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)

models_cfg = [
    ('DecisionTree', DecisionTreeRegressor(), {
        'max_depth':         [5, 10, 20, None],
        'min_samples_split': [2, 10],
        'min_samples_leaf':  [1, 4],
        'random_state':      [42],
    }),
    ('RandomForest', RandomForestRegressor(n_jobs=-1), {
        'n_estimators':      [100, 200],
        'max_depth':         [10, 20, None],
        'min_samples_split': [2, 10],
        'min_samples_leaf':  [1, 4],
        'random_state':      [42],
    }),
]

predictions = {}
metrics     = {}
best_models = {}

for name, model, params in models_cfg:
    print(f"  Training {name}...")
    gs = GridSearchCV(model, params, cv=tscv,
                      scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    best_models[name] = best

    pred = pd.Series(best.predict(X_test), index=test.index)
    predictions[name] = pred

    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
    metrics[name] = {'MAE': round(mae,3), 'RMSE': round(rmse,3), 'MAPE(%)': round(mape,2)}
    print(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.2f}%")

# ── 05 PREDICTIONS COMPARISON ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.patch.set_facecolor(DARK)
fig.suptitle('Model Predictions vs Real Price  (Test Set — last 252 trading days)',
             fontsize=14, color=WHITE, fontweight='bold', y=1.01)

colors_m = {'DecisionTree': BLUE, 'RandomForest': GREEN}

for ax, (name, pred) in zip(axes, predictions.items()):
    # last 60 days of train for context
    ax.plot(train['Price'].tail(60).index, train['Price'].tail(60),
            color=GRAY, linewidth=1, alpha=0.6, label='Train (last 60d)')
    ax.plot(test.index, y_test, color=ACCENT, linewidth=2, label='Real Price')
    ax.plot(test.index, pred,   color=colors_m[name], linewidth=1.6,
            linestyle='--', label=name)
    m = metrics[name]
    ax.set_title(f"{name}  |  MAE: ${m['MAE']}  |  RMSE: ${m['RMSE']}  |  MAPE: {m['MAPE(%)']}%",
                 color=WHITE, fontsize=11)
    ax.set_ylabel('USD/barrel', color=GRAY)
    ax.legend(fontsize=9, loc='upper left')
    ax.yaxis.grid(True); ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f'{IMG}/05_predictions.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("✔ 05_predictions.png")

# ── 06 METRICS COMPARISON ─────────────────────────────────────────────────────
df_met = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Model'})

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.patch.set_facecolor(DARK)
fig.suptitle('Metrics Comparison — Decision Tree vs Random Forest',
             fontsize=14, color=WHITE, fontweight='bold')

metric_colors = [BLUE, ACCENT, GREEN]
for ax, col, c in zip(axes, ['MAE', 'RMSE', 'MAPE(%)'], metric_colors):
    bars = ax.bar(df_met['Model'], df_met[col], color=c, alpha=0.85, width=0.45)
    ax.set_title(col, color=WHITE, fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', color=GRAY)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    for bar, v in zip(bars, df_met[col]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                str(v), ha='center', va='bottom', fontsize=11, color=WHITE, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{IMG}/06_metrics.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("✔ 06_metrics.png")

# ── 07 FEATURE IMPORTANCE ─────────────────────────────────────────────────────
rf_model = best_models['RandomForest']
fi = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(DARK)
bar_colors = [ACCENT if v > fi.quantile(0.7) else BLUE for v in fi.values]
ax.barh(fi.index, fi.values, color=bar_colors, alpha=0.88)
ax.set_title('Feature Importance — Random Forest  (Top 15)', fontsize=14, color=WHITE, fontweight='bold', pad=12)
ax.set_xlabel('Importance', color=GRAY, fontsize=10)
ax.xaxis.grid(True); ax.set_axisbelow(True)
for i, v in enumerate(fi.values):
    ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=8.5, color=WHITE)
plt.tight_layout()
plt.savefig(f'{IMG}/07_feature_importance.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("✔ 07_feature_importance.png")

# ── 08 RESIDUALS ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.patch.set_facecolor(DARK)
fig.suptitle('Residual Analysis', fontsize=14, color=WHITE, fontweight='bold')

for ax, (name, pred), c in zip(axes, predictions.items(), [BLUE, GREEN]):
    residuals = y_test - pred
    ax.plot(test.index, residuals, color=c, linewidth=0.9, alpha=0.8)
    ax.axhline(0, color=WHITE, linewidth=1, linestyle='--', alpha=0.5)
    ax.fill_between(test.index, residuals, alpha=0.15, color=c)
    ax.set_title(f'Residuals — {name}', color=WHITE, fontsize=11)
    ax.set_ylabel('Error (Real − Predicted)', color=GRAY)
    ax.yaxis.grid(True); ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f'{IMG}/08_residuals.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("✔ 08_residuals.png")

print("\n✅ All images generated successfully.")
