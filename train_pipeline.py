import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.optimize import minimize
from lightgbm import LGBMRegressor
from lightgbm import early_stopping
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor


# ---------------------------------------------------------------------------
#  Actigraphy helpers (time-series → summary statistics per participant)
# ---------------------------------------------------------------------------

def extract_actigraphy_features(filepath):
    try:
        df = pd.read_parquet(filepath)
    except Exception:
        return None

    features = {}

    # Movement intensity
    enmo = df['enmo'].dropna()
    if len(enmo) > 0:
        features['acti_enmo_mean'] = enmo.mean()
        features['acti_enmo_std'] = enmo.std()
        features['acti_enmo_median'] = enmo.median()
        features['acti_enmo_q25'] = enmo.quantile(0.25)
        features['acti_enmo_q75'] = enmo.quantile(0.75)
        features['acti_enmo_max'] = enmo.max()
        features['acti_active_pct'] = (enmo > 0.03).mean()
        features['acti_sedentary_pct'] = (enmo < 0.01).mean()

    # Body posture
    anglez = df['anglez'].dropna()
    if len(anglez) > 0:
        features['acti_anglez_mean'] = anglez.mean()
        features['acti_anglez_std'] = anglez.std()
        features['acti_anglez_median'] = anglez.median()

    # Device wear compliance
    if 'non-wear_flag' in df.columns:
        features['acti_nonwear_pct'] = df['non-wear_flag'].mean()

    # Ambient light (indoor vs outdoor proxy)
    light = df['light'].dropna()
    if len(light) > 0:
        features['acti_light_mean'] = light.mean()

    # Recording duration
    features['acti_n_steps'] = len(df)

    return features


def load_actigraphy_features(ids, series_dir):
    rows = []
    for pid in ids:
        filepath = os.path.join(series_dir, f'id={pid}', 'part-0.parquet')
        if os.path.exists(filepath):
            feats = extract_actigraphy_features(filepath)
            if feats is not None:
                feats['id'] = pid
                rows.append(feats)
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=['id'])

# KAGGLE PATHS
TRAIN_PATH = "/kaggle/input/child-mind-institute-problematic-internet-use/train.csv"
TEST_PATH  = "/kaggle/input/child-mind-institute-problematic-internet-use/test.csv"

for p in [TRAIN_PATH, TEST_PATH]:
    assert os.path.exists(p), f'File not found: {p}'

# Load datasets
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("="*70)
print("ORIGINAL DATA")
print("="*70)
print(f"Train: {train.shape}, Test: {test.shape}")

test_ids = test['id'].copy()

# ---------------------------------------------------------------------------
#  Target - use PCIAT_Total (continuous) instead of sii (categorical)
# ---------------------------------------------------------------------------

train = train[~train['sii'].isna()].copy()

y_pciat = train['PCIAT-PCIAT_Total'].values 
y_sii = train['sii'].astype(int).values      # Keep for stratification & validation

train = train.drop(columns=['sii', 'PCIAT-PCIAT_Total']) 

# ---------------------------------------------------------------------------
#  Actigraphy - merge accelerometer summary stats (~25% have data, rest → NaN)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(TRAIN_PATH)
SERIES_TRAIN_DIR = os.path.join(BASE_DIR, 'series_train.parquet')
SERIES_TEST_DIR = os.path.join(BASE_DIR, 'series_test.parquet')

acti_train = load_actigraphy_features(train['id'].values, SERIES_TRAIN_DIR)
acti_test = load_actigraphy_features(test['id'].values, SERIES_TEST_DIR)

train = train.merge(acti_train, on='id', how='left')
test = test.merge(acti_test, on='id', how='left')

acti_cols = [c for c in acti_train.columns if c != 'id']
print(f"\nActigraphy: {len(acti_train)} train / {len(acti_test)} test participants, {len(acti_cols)} features")

# Remove ID
if 'id' in train.columns:
    train = train.drop(columns=['id'])
if 'id' in test.columns:
    test = test.drop(columns=['id'])

# Use only columns that exist in BOTH train and test
train_cols = set(train.columns)
test_cols = set(test.columns)
common_cols = sorted(list(train_cols & test_cols))

print(f"\n{'='*70}")
print("COLUMN FILTERING")
print(f"{'='*70}")
print(f"Train columns: {len(train_cols)}")
print(f"Test columns: {len(test_cols)}")
print(f"Common columns: {len(common_cols)}")

# Only common columns
train = train[common_cols]
test = test[common_cols]

print(f"\nAfter filtering to common columns:")
print(f"  Train: {train.shape}")
print(f"  Test: {test.shape}")

# Check what we're removing
removed_cols = train_cols - test_cols
if removed_cols:
    print(f"\nRemoved {len(removed_cols)} columns only in train:")
    pciat_cols = [c for c in removed_cols if 'PCIAT' in c]
    print(f"  PCIAT columns: {len(pciat_cols)}")
    if pciat_cols:
        print(f"  Examples: {pciat_cols[:5]}")

# Feature processing
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
num_cols = [c for c in train.columns if c not in cat_cols]

print(f"\n{'='*70}")
print("FEATURES")
print(f"{'='*70}")
print(f"Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}")

# Handle categorical features
for col in cat_cols:
    combined = pd.concat([
        train[col].fillna('missing'),
        test[col].fillna('missing')
    ])
    categories = combined.unique()
    cat_map = {cat: i for i, cat in enumerate(categories)}

    train[col] = train[col].fillna('missing').map(cat_map)
    test[col] = test[col].fillna('missing').map(cat_map)

# ---------------------------------------------------------------------------
#  Imputation - IterativeImputer with LightGBM (MICE-style, non-linear)
# ---------------------------------------------------------------------------
imputer = IterativeImputer(
    estimator=LGBMRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        verbose=-1,
        n_jobs=-1
    ),
    max_iter=5,
    random_state=42,
    verbose=0
)
train_values = imputer.fit_transform(train)
test_values = imputer.transform(test)

print(f"\nAfter imputation:")
print(f"  Train NaN: {np.isnan(train_values).sum()}")
print(f"  Test NaN: {np.isnan(test_values).sum()}")

# Add simple feature engineering
def add_basic_features(data):
    """Add simple aggregated features"""
    df = pd.DataFrame(data)

    # Basic row statistics
    row_mean = df.mean(axis=1).values.reshape(-1, 1)
    row_std = df.std(axis=1).values.reshape(-1, 1)
    row_min = df.min(axis=1).values.reshape(-1, 1)
    row_max = df.max(axis=1).values.reshape(-1, 1)

    # Combine
    return np.hstack([data, row_mean, row_std, row_min, row_max])

train_values = add_basic_features(train_values)
test_values = add_basic_features(test_values)

feature_names = list(common_cols) + ['row_mean', 'row_std', 'row_min', 'row_max']

print(f"\nAfter feature engineering: {train_values.shape}")

# Model training
SEED = 42
N_SPLITS = 10

print(f"\n{'='*70}")
print("MODEL TRAINING")
print(f"{'='*70}")

# ---------------------------------------------------------------------------
#  Model definitions - LGBM, XGBoost, CatBoost, ExtraTrees
# ---------------------------------------------------------------------------

def train_lgbm(X_train, y_train, X_val, y_val, seed=42):
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        verbose=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
    )
    return model

def train_xgb(X_train, y_train, X_val, y_val, seed=42):
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        verbosity=0,
        early_stopping_rounds=50
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model

def train_catboost(X_train, y_train, X_val, y_val, seed=42):
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        subsample=0.8,
        reg_lambda=0.1,
        random_state=seed,
        verbose=0,
        early_stopping_rounds=50
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model

def train_extratrees(X_train, y_train, X_val, y_val, seed=42):
    model = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

# ---------------------------------------------------------------------------
#  Cross-validation - stratify on sii, train on PCIAT_Total
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# OOF predictions per model
oof_lgbm = np.zeros(len(train_values))
oof_xgb = np.zeros(len(train_values))
oof_cat = np.zeros(len(train_values))
oof_et = np.zeros(len(train_values))

# Test predictions per model
test_lgbm = np.zeros(len(test_values))
test_xgb = np.zeros(len(test_values))
test_cat = np.zeros(len(test_values))
test_et = np.zeros(len(test_values))

# Feature importance (averaged across folds from LGBM)
feat_importance = np.zeros(len(feature_names))

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_values, y_sii)):
    print(f"\nFold {fold+1}/{N_SPLITS}:")

    X_tr = train_values[tr_idx]
    X_val = train_values[val_idx]
    y_tr = y_pciat[tr_idx] 
    y_val = y_pciat[val_idx] 

    lgbm = train_lgbm(X_tr, y_tr, X_val, y_val, seed=SEED + fold)
    xgb = train_xgb(X_tr, y_tr, X_val, y_val, seed=SEED + fold)
    cat = train_catboost(X_tr, y_tr, X_val, y_val, seed=SEED + fold)
    et = train_extratrees(X_tr, y_tr, X_val, y_val, seed=SEED + fold)
    
    # Store individual model predictions
    oof_lgbm[val_idx] = lgbm.predict(X_val)
    oof_xgb[val_idx] = xgb.predict(X_val)
    oof_cat[val_idx] = cat.predict(X_val)
    oof_et[val_idx] = et.predict(X_val)
    
    test_lgbm += lgbm.predict(test_values) / N_SPLITS
    test_xgb += xgb.predict(test_values) / N_SPLITS
    test_cat += cat.predict(test_values) / N_SPLITS
    test_et += et.predict(test_values) / N_SPLITS

    feat_importance += lgbm.feature_importances_ / N_SPLITS

    print(f"  LGBM val mean: {oof_lgbm[val_idx].mean():.2f}")
    print(f"  XGB val mean: {oof_xgb[val_idx].mean():.2f}")
    print(f"  CAT val mean: {oof_cat[val_idx].mean():.2f}")
    print(f"  ET val mean: {oof_et[val_idx].mean():.2f}")

# ---------------------------------------------------------------------------
#  Ensemble weight optimization - maximize QWK via Nelder-Mead
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("OPTIMIZING ENSEMBLE WEIGHTS")
print(f"{'='*70}")

def optimize_weights(oof_preds_list, y_true, apply_thresholds_fn, init_thr=[30, 49, 79]):
    """
    Find optimal weights for ensemble that maximize QWK.
    """
    from scipy.optimize import minimize
    
    def objective(weights):
        # Normalize weights to sum to 1
        w = np.abs(weights)
        w = w / w.sum()
        
        # Combine predictions
        combined = sum(w[i] * oof_preds_list[i] for i in range(len(oof_preds_list)))
        
        # Apply thresholds and compute QWK
        preds_class = apply_thresholds_fn(combined, init_thr)
        qwk = cohen_kappa_score(y_true, preds_class, weights="quadratic")
        
        return -qwk  # Minimize negative QWK
    
    # Start with equal weights
    init_weights = np.ones(len(oof_preds_list)) / len(oof_preds_list)
    
    # Optimize
    result = minimize(objective, init_weights, method='Nelder-Mead')
    
    # Normalize final weights
    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / optimal_weights.sum()
    
    return optimal_weights

def apply_thresholds_simple(preds, thr):
    cuts = [-np.inf] + list(thr) + [np.inf]
    return np.clip(np.digitize(preds, cuts) - 1, 0, 3)

# Optimize weights
oof_preds_list = [oof_lgbm, oof_xgb, oof_cat, oof_et]
optimal_weights = optimize_weights(oof_preds_list, y_sii, apply_thresholds_simple)

print(f"Optimal weights:")
print(f"  LGBM: {optimal_weights[0]:.3f}")
print(f"  XGB:  {optimal_weights[1]:.3f}")
print(f"  CAT:  {optimal_weights[2]:.3f}")
print(f"  ET:   {optimal_weights[3]:.3f}")

# Apply optimal weights
oof_preds = (
    optimal_weights[0] * oof_lgbm +
    optimal_weights[1] * oof_xgb +
    optimal_weights[2] * oof_cat +
    optimal_weights[3] * oof_et
)

test_preds = (
    optimal_weights[0] * test_lgbm +
    optimal_weights[1] * test_xgb +
    optimal_weights[2] * test_cat +
    optimal_weights[3] * test_et
)

print(f"\n{'='*70}")
print("PREDICTIONS SUMMARY")
print(f"{'='*70}")
print(f"OOF: min={oof_preds.min():.3f}, max={oof_preds.max():.3f}, mean={oof_preds.mean():.3f}")
print(f"Test: min={test_preds.min():.3f}, max={test_preds.max():.3f}, mean={test_preds.mean():.3f}")

# ---------------------------------------------------------------------------
#  Threshold optimization - PCIAT scores → SII classes (0-3)
# ---------------------------------------------------------------------------
def optimize_thresholds(y_true, y_pred):
    def loss(thr):
        thr = np.sort(thr)
        cuts = [-np.inf] + list(thr) + [np.inf]
        y_hat = np.clip(np.digitize(y_pred, cuts) - 1, 0, 3)
        return -cohen_kappa_score(y_true, y_hat, weights="quadratic")

    res = minimize(loss, x0=[30, 49, 79], method="nelder-mead")
    return np.sort(res.x)

def apply_thresholds(preds, thr):
    cuts = [-np.inf] + list(thr) + [np.inf]
    return np.clip(np.digitize(preds, cuts) - 1, 0, 3)

thresholds = optimize_thresholds(y_sii, oof_preds)
print(f"\nOptimized thresholds: {thresholds}")

oof_classes = apply_thresholds(oof_preds, thresholds)
test_classes = apply_thresholds(test_preds, thresholds)

qwk = cohen_kappa_score(y_sii, oof_classes, weights="quadratic")
print(f"\n{'='*70}")
print(f"OOF Quadratic Weighted Kappa: {qwk:.4f}")
print(f"{'='*70}")

# Check distributions
print(f"\nTarget distribution (sii):")
print(pd.Series(y_sii).value_counts().sort_index())

print(f"\nOOF prediction distribution:")
print(pd.Series(oof_classes).value_counts().sort_index())

print(f"\nTest prediction distribution:")
test_dist = pd.Series(test_classes).value_counts().sort_index()
print(test_dist)

# Create submission
submission = pd.DataFrame({
    "id": test_ids,
    "sii": test_classes.astype(int)
})

print(f"\n{'='*70}")
print("SUBMISSION")
print(f"{'='*70}")
print(submission)

submission.to_csv("submission.csv", index=False)

# ---------------------------------------------------------------------------
#  Data visualizations
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Consistent style for all graphs
plt.style.use('seaborn-v0_8-whitegrid')
COLORS_SII = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']  # Green, Yellow, Orange, Red

# Reload original data for visualization
train_orig = pd.read_csv(TRAIN_PATH)

# -----------------------------------------------------------------------------
# 1. TARGET DISTRIBUTION (SII) - Shows class imbalance
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
sii_counts = pd.Series(y_sii).value_counts().sort_index()
bars = plt.bar(sii_counts.index, sii_counts.values, color=COLORS_SII, edgecolor='black', linewidth=1.5)
plt.xlabel('SII Class', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.title('Target Distribution: Severity Impairment Index (SII)', fontsize=14, fontweight='bold')
plt.xticks([0, 1, 2, 3], ['0 (None)\n58%', '1 (Mild)\n27%', '2 (Moderate)\n14%', '3 (Severe)\n1.2%'])
for bar, count in zip(bars, sii_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{count}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 2. PCIAT TOTAL DISTRIBUTION - Continuous score with SII thresholds
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.hist(y_pciat[~np.isnan(y_pciat)], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
plt.axvline(30, color='#2ecc71', linestyle='--', linewidth=2, label='SII=1 threshold (30)')
plt.axvline(49, color='#f1c40f', linestyle='--', linewidth=2, label='SII=2 threshold (49)')
plt.axvline(79, color='#e74c3c', linestyle='--', linewidth=2, label='SII=3 threshold (79)')
plt.xlabel('PCIAT Total Score', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('PCIAT Total Score Distribution (0-100)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 3. MISSING DATA - Top 20 features with most missing values
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
missing = train.isna().mean().sort_values(ascending=False).head(20)
colors_missing = ['#e74c3c' if x > 0.7 else '#f39c12' if x > 0.4 else '#3498db' for x in missing.values]
bars = plt.barh(range(len(missing)), missing.values * 100, color=colors_missing, edgecolor='black')
plt.yticks(range(len(missing)), missing.index, fontsize=9)
plt.xlabel('Missing Percentage (%)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 20 Features with Most Missing Values', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.axvline(50, color='gray', linestyle='--', alpha=0.5)
for bar, pct in zip(bars, missing.values):
    plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{pct*100:.0f}%', va='center', fontsize=9)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 4. FEATURE IMPORTANCE - Top 20 most predictive features (LGBM, averaged across folds)
# -----------------------------------------------------------------------------
fi = pd.Series(feat_importance, index=feature_names).sort_values(ascending=False).head(20)
plt.figure(figsize=(10, 6))
colors_fi = ['#e74c3c' if name.startswith('acti_') else '#3498db' for name in fi.index]
bars = plt.barh(range(len(fi)), fi.values, color=colors_fi, edgecolor='black')
plt.yticks(range(len(fi)), fi.index, fontsize=9)
plt.xlabel('Importance (split count, averaged across folds)', fontsize=12)
plt.title('Top 20 Feature Importances (LightGBM)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
import matplotlib.patches as mpatches
plt.legend(handles=[
    mpatches.Patch(color='#3498db', label='Tabular features'),
    mpatches.Patch(color='#e74c3c', label='Actigraphy features')
], fontsize=10, loc='lower right')
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 5. DATA AVAILABILITY - Samples with vs without target
# -----------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
total_samples = len(train_orig)
has_sii = train_orig['sii'].notna().sum()
has_no_sii = total_samples - has_sii
bars = plt.bar(['With Target\n(can train)', 'Without Target\n(cannot train)'], [has_sii, has_no_sii], 
               color=['#27ae60', '#e74c3c'], edgecolor='black', linewidth=1.5)
plt.ylabel('Number of Samples', fontsize=12)
plt.title(f'Data Availability: {has_no_sii} samples ({100*has_no_sii/total_samples:.1f}%) have no target', fontsize=14, fontweight='bold')
for bar, val in zip(bars, [has_sii, has_no_sii]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f'{val}', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
