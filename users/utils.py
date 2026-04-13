"""
users/utils.py
ML logic for WESAD stress detection — keeps views.py clean.
Two-stage XGBoost pipeline:
  Stage A : Stress vs Non-Stress  (binary)
  Stage B : Baseline / Amusement / Meditation  (3-class, non-stress only)
"""

import os
import io
import base64
import pickle
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

FS_TARGET  = 4                      # all signals resampled to 4 Hz
LABEL_MAP  = {1: 0, 2: 1, 3: 2, 4: 3}   # raw WESAD labels → 0-based
CLASS_NAMES = ['Baseline', 'Stress', 'Amusement', 'Meditation']

# ─────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────

def _model_dir(base_dir):
    return os.path.join(base_dir, 'media', 'models')

def _stage_a_path(base_dir):
    return os.path.join(_model_dir(base_dir), 'stageA_stress_vs_nonstress.pkl')

def _stage_b_path(base_dir):
    return os.path.join(_model_dir(base_dir), 'stageB_nonstress_3class.pkl')

def _feature_cols_path(base_dir):
    return os.path.join(_model_dir(base_dir), 'feature_columns.pkl')

def _dataset_dir(base_dir):
    """Root folder that contains WESAD subject sub-dirs (S2, S3, …)."""
    return os.path.join(base_dir, 'media', 'WESAD')


# ─────────────────────────────────────────────
# Shared plot helpers
# ─────────────────────────────────────────────

def _b64(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, facecolor='#0C0A09', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _plot_confusion_matrix(y_true, y_pred, class_names):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='#0C0A09')
    ax.set_facecolor('#1A1714')
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names,
                cmap='YlOrBr', ax=ax, linewidths=0.5, linecolor='#0C0A09')
    ax.set_xlabel('Predicted', color='#A8A29E')
    ax.set_ylabel('Actual',    color='#A8A29E')
    ax.set_title('Confusion Matrix', color='#FFFBEB', fontsize=13)
    ax.tick_params(colors='#A8A29E')
    return _b64(fig)


def _plot_class_accuracy(y_true, y_pred, class_names):
    from sklearn.metrics import confusion_matrix
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    row_sums = cm.sum(axis=1)
    acc = np.where(row_sums > 0, cm.diagonal() / row_sums, 0.0)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0C0A09')
    ax.set_facecolor('#1A1714')
    colors = ['#22C55E' if a >= 0.8 else '#F59E0B' if a >= 0.6 else '#EF4444' for a in acc]
    bars   = ax.bar(class_names, acc * 100, color=colors, edgecolor='#0C0A09', linewidth=0.5)
    ax.set_ylim(0, 115)
    ax.set_ylabel('Accuracy %', color='#A8A29E')
    ax.set_title('Per-Class Accuracy', color='#FFFBEB', fontsize=13)
    ax.tick_params(colors='#A8A29E')
    ax.spines[:].set_color('#2A2520')
    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val*100:.1f}%', ha='center', va='bottom',
                color='#FFFBEB', fontsize=9, fontfamily='monospace')
    plt.tight_layout()
    return _b64(fig)


def _plot_prediction_distribution(y_pred, class_names):
    counts = np.bincount(y_pred, minlength=len(class_names))
    colors = ['#F59E0B', '#EF4444', '#60A5FA', '#22C55E'][:len(class_names)]
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#0C0A09')
    ax.set_facecolor('#0C0A09')
    wedges, texts, autotexts = ax.pie(
        counts, labels=class_names, autopct='%1.1f%%',
        colors=colors, startangle=140,
        wedgeprops=dict(edgecolor='#0C0A09', linewidth=1.5))
    for t in texts:     t.set_color('#A8A29E'); t.set_fontsize(9)
    for t in autotexts: t.set_color('#0C0A09'); t.set_fontsize(9); t.set_fontweight('bold')
    ax.set_title('Prediction Distribution', color='#FFFBEB', fontsize=13)
    plt.tight_layout()
    return _b64(fig)


def _plot_training_accuracy(history_dict):
    """Bar chart of per-stage train accuracy."""
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0C0A09')
    ax.set_facecolor('#1A1714')
    ax.spines[:].set_color('#2A2520')
    labels = list(history_dict.keys())
    vals   = list(history_dict.values())
    colors = ['#F59E0B', '#60A5FA']
    bars   = ax.bar(labels, [v * 100 for v in vals], color=colors, edgecolor='#0C0A09')
    ax.set_ylim(0, 115)
    ax.set_ylabel('Accuracy %', color='#A8A29E')
    ax.set_title('Stage Training Accuracy', color='#FFFBEB', fontsize=13)
    ax.tick_params(colors='#A8A29E')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val*100:.1f}%', ha='center', va='bottom',
                color='#FFFBEB', fontsize=10, fontfamily='monospace')
    plt.tight_layout()
    return _b64(fig)


def _build_per_class_metrics(y_true, y_pred, class_names):
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0)
    return [{'name':      class_names[i],
             'precision': f"{prec[i]*100:.1f}%",
             'recall':    f"{rec[i]*100:.1f}%",
             'f1':        f"{f1[i]*100:.1f}%",
             'support':   int(sup[i])}
            for i in range(len(class_names))]


# ─────────────────────────────────────────────
# Signal / Feature helpers  (same as Colab)
# ─────────────────────────────────────────────
def _split(X, y, groups, test_size=0.2):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    n_subjects = len(np.unique(groups))
    if n_subjects >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
    else:
        all_idx = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            all_idx, test_size=test_size, random_state=42, shuffle=True)
    return train_idx, test_idx

def _majority_label(arr):
    return Counter(arr).most_common(1)[0][0]


def _downsample_mean(x, fs_original, fs_target=FS_TARGET):
    x = np.asarray(x)
    if fs_original == fs_target:
        return x
    factor = int(fs_original // fs_target)
    n = (len(x) // factor) * factor
    return x[:n].reshape(-1, factor).mean(axis=1)


def _downsample_labels_mode(lbls, fs_original, fs_target=FS_TARGET):
    lbls = np.asarray(lbls)
    if fs_original == fs_target:
        return lbls
    factor = int(fs_original // fs_target)
    n = (len(lbls) // factor) * factor
    lbls = lbls[:n]
    return np.array([_majority_label(lbls[i:i+factor].tolist())
                     for i in range(0, n, factor)])


def _fft_features(x, prefix, fs=4.0):
    x = np.asarray(x, dtype=np.float32) - np.mean(x)
    n = len(x)
    if n < 8:
        return {f"{prefix}_dom_freq": 0.0, f"{prefix}_spec_energy": 0.0}
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag   = np.abs(np.fft.rfft(x))
    dom_freq = float(freqs[int(np.argmax(mag[1:]) + 1)]) if len(mag) > 1 else 0.0
    return {f"{prefix}_dom_freq":    dom_freq,
            f"{prefix}_spec_energy": float(np.sum(mag ** 2) / n)}


def _extract_stats(x, prefix, fs=4.0, use_fft=False):
    x = np.asarray(x, dtype=np.float32)
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    t     = np.arange(len(x), dtype=np.float32)
    slope = float(np.polyfit(t, x, 1)[0])
    feats = {
        f"{prefix}_mean":   float(np.mean(x)),
        f"{prefix}_std":    float(np.std(x)),
        f"{prefix}_min":    float(np.min(x)),
        f"{prefix}_max":    float(np.max(x)),
        f"{prefix}_range":  float(np.max(x) - np.min(x)),
        f"{prefix}_median": float(q50),
        f"{prefix}_iqr":    float(q75 - q25),
        f"{prefix}_slope":  slope,
        f"{prefix}_energy": float(np.mean(x ** 2)),
    }
    if use_fft:
        feats.update(_fft_features(x, prefix, fs=fs))
    return feats


def _subject_to_rows(data, subject_id, window_sec=60, step_sec=15):
    """Convert one subject's raw dict → list of feature-row dicts."""
    w = data["signal"]["wrist"]
    c = data["signal"]["chest"]
    labels = np.array(data["label"]).astype(int)

    # Wrist
    W_EDA   = np.squeeze(np.array(w["EDA"]))
    W_TEMP  = np.squeeze(np.array(w["TEMP"]))
    W_BVP_4 = _downsample_mean(np.squeeze(np.array(w["BVP"])), 64)
    W_ACC   = np.array(w["ACC"])
    W_Ax    = _downsample_mean(W_ACC[:, 0], 32)
    W_Ay    = _downsample_mean(W_ACC[:, 1], 32)
    W_Az    = _downsample_mean(W_ACC[:, 2], 32)

    # Chest
    C_ECG   = _downsample_mean(np.squeeze(np.array(c["ECG"])),  700)
    C_EMG   = _downsample_mean(np.squeeze(np.array(c["EMG"])),  700)
    C_RESP  = _downsample_mean(np.squeeze(np.array(c["Resp"])), 700)
    C_EDA   = _downsample_mean(np.squeeze(np.array(c["EDA"])),  700)
    C_TEMP  = _downsample_mean(np.squeeze(np.array(c["Temp"])), 700)
    C_ACC   = np.array(c["ACC"])
    C_Ax    = _downsample_mean(C_ACC[:, 0], 700)
    C_Ay    = _downsample_mean(C_ACC[:, 1], 700)
    C_Az    = _downsample_mean(C_ACC[:, 2], 700)
    labels_4 = _downsample_labels_mode(labels, 700)

    L = min(len(labels_4), len(W_EDA), len(W_TEMP), len(W_BVP_4),
            len(W_Ax), len(W_Ay), len(W_Az),
            len(C_ECG), len(C_EMG), len(C_RESP), len(C_EDA), len(C_TEMP),
            len(C_Ax), len(C_Ay), len(C_Az))

    signals = dict(labels_4=labels_4[:L],
                   W_EDA=W_EDA[:L], W_TEMP=W_TEMP[:L], W_BVP=W_BVP_4[:L],
                   W_Ax=W_Ax[:L], W_Ay=W_Ay[:L], W_Az=W_Az[:L],
                   C_ECG=C_ECG[:L], C_EMG=C_EMG[:L], C_RESP=C_RESP[:L],
                   C_EDA=C_EDA[:L], C_TEMP=C_TEMP[:L],
                   C_Ax=C_Ax[:L], C_Ay=C_Ay[:L], C_Az=C_Az[:L])

    win  = window_sec * FS_TARGET
    step = step_sec   * FS_TARGET
    rows = []

    for start in range(0, L - win + 1, step):
        end     = start + win
        seg_lbl = signals['labels_4'][start:end]
        if not np.all(np.isin(seg_lbl, list(LABEL_MAP.keys()))):
            continue
        y_raw = _majority_label(seg_lbl.tolist())
        y     = LABEL_MAP[y_raw]

        f = {}
        f.update(_extract_stats(signals['W_EDA'][start:end],  "W_EDA",  use_fft=False))
        f.update(_extract_stats(signals['W_TEMP'][start:end], "W_TEMP", use_fft=False))
        f.update(_extract_stats(signals['W_BVP'][start:end],  "W_BVP",  use_fft=True))
        f.update(_extract_stats(signals['W_Ax'][start:end],   "W_ACCx", use_fft=True))
        f.update(_extract_stats(signals['W_Ay'][start:end],   "W_ACCy", use_fft=True))
        f.update(_extract_stats(signals['W_Az'][start:end],   "W_ACCz", use_fft=True))
        f.update(_extract_stats(signals['C_ECG'][start:end],  "C_ECG",  use_fft=True))
        f.update(_extract_stats(signals['C_RESP'][start:end], "C_RESP", use_fft=True))
        f.update(_extract_stats(signals['C_EMG'][start:end],  "C_EMG",  use_fft=False))
        f.update(_extract_stats(signals['C_EDA'][start:end],  "C_EDA",  use_fft=False))
        f.update(_extract_stats(signals['C_TEMP'][start:end], "C_TEMP", use_fft=False))
        f.update(_extract_stats(signals['C_Ax'][start:end],   "C_ACCx", use_fft=True))
        f.update(_extract_stats(signals['C_Ay'][start:end],   "C_ACCy", use_fft=True))
        f.update(_extract_stats(signals['C_Az'][start:end],   "C_ACCz", use_fft=True))

        f["subject"] = subject_id
        f["label"]   = y
        rows.append(f)

    return rows


def _load_dataset(base_dir, window_sec=60, step_sec=15):
    """Load all WESAD subjects from BASE_DIR/media/WESAD/ and build feature DataFrame."""
    import pandas as pd
    dataset_root = _dataset_dir(base_dir)
    pkl_files    = sorted(glob.glob(os.path.join(dataset_root, '**', 'S*.pkl'), recursive=True))
    if not pkl_files:
        raise FileNotFoundError(
            f"No WESAD .pkl files found under:\n{dataset_root}\n\n"
            "Expected layout:  BASE_DIR/media/WESAD/S2/S2.pkl  (etc.)")

    all_rows = []
    for p in pkl_files:
        sid = os.path.basename(os.path.dirname(p))
        with open(p, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        all_rows.extend(_subject_to_rows(data, sid, window_sec, step_sec))

    df = pd.DataFrame(all_rows)
    return df


# ─────────────────────────────────────────────
# Two-stage predict helper
# ─────────────────────────────────────────────

def _two_stage_predict(modelA, modelB, X, thr=0.35):
    inv_mapB = {0: 0, 1: 2, 2: 3}
    predA    = (modelA.predict_proba(X)[:, 1] > thr).astype(int)
    predB    = modelB.predict(X).astype(int)
    final    = np.where(predA == 1, 1, np.vectorize(inv_mapB.get)(predB))
    return final


# ─────────────────────────────────────────────
# Evaluate saved models
# ─────────────────────────────────────────────

def quick_evaluate(base_dir):
    """Load saved models → evaluate on a held-out test split."""
    import joblib, pandas as pd
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import f1_score, classification_report, accuracy_score

    for path, label in [(_stage_a_path(base_dir),    'Stage A model'),
                         (_stage_b_path(base_dir),    'Stage B model'),
                         (_feature_cols_path(base_dir), 'Feature column list')]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found at:\n{path}\n\n"
                "Run Full Retrain first, or copy your trained .pkl files to "
                "BASE_DIR/media/stress_models/")

    modelA       = joblib.load(_stage_a_path(base_dir))
    modelB       = joblib.load(_stage_b_path(base_dir))
    feature_cols = joblib.load(_feature_cols_path(base_dir))

    df     = _load_dataset(base_dir)
    X      = df[feature_cols]
    y      = df['label'].values
    groups = df['subject'].values

    _, test_idx = _split(X, y, groups, test_size=0.2)
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    y_pred   = _two_stage_predict(modelA, modelB, X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    acc      = accuracy_score(y_test, y_pred)

    correct_mask = y_test == y_pred

    return {
        'mode':                  'quick',
        'test_accuracy':         f"{acc*100:.1f}%",
        'macro_f1':              f"{macro_f1*100:.1f}%",
        'total_samples':         len(y_test),
        'correct_predictions':   int(correct_mask.sum()),
        'wrong_predictions':     int((~correct_mask).sum()),
        'classification_report': classification_report(y_test, y_pred, target_names=CLASS_NAMES),
        'per_class_metrics':     _build_per_class_metrics(y_test, y_pred, CLASS_NAMES),
        'cm_base64':             _plot_confusion_matrix(y_test, y_pred, CLASS_NAMES),
        'class_acc_base64':      _plot_class_accuracy(y_test, y_pred, CLASS_NAMES),
        'dist_base64':           _plot_prediction_distribution(y_pred, CLASS_NAMES),
        'class_names':           CLASS_NAMES,
    }


# ─────────────────────────────────────────────
# Full Retrain
# ─────────────────────────────────────────────

def full_retrain(base_dir,
                 window_sec=60, step_sec=15,
                 n_estimators_a=700, n_estimators_b=800,
                 stress_threshold=0.35):
    """Train both XGBoost stages from scratch and save models."""
    import joblib, pandas as pd
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.utils import resample as sk_resample
    from sklearn.metrics import f1_score, classification_report, accuracy_score
    from xgboost import XGBClassifier

    os.makedirs(_model_dir(base_dir), exist_ok=True)

    # ── 1. Build feature matrix ──────────────────────────────────────────
    df     = _load_dataset(base_dir, window_sec, step_sec)
    X_all  = df.drop(columns=['label', 'subject'])
    y_all  = df['label'].values
    groups = df['subject'].values

    feature_cols = list(X_all.columns)
    joblib.dump(feature_cols, _feature_cols_path(base_dir))

    # ── 2. Subject-level train/test split ───────────────────────────────
    n_subjects = len(np.unique(groups))

    if n_subjects >= 2:
        # Normal subject-level split — keeps subjects separate between train/test
        train_idx, test_idx = _split(X_all, y_all, groups, test_size=0.2)
    else:
        # Only 1 subject — fall back to a simple 80/20 row-level split
        from sklearn.model_selection import train_test_split
        all_idx = np.arange(len(X_all))
        train_idx, test_idx = train_test_split(
            all_idx, test_size=0.2, random_state=42, shuffle=True)

    X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
    y_train, y_test = y_all[train_idx],      y_all[test_idx]

    # ── 3. Stage A: Stress vs Non-Stress ────────────────────────────────
    yA_train = (y_train == 1).astype(int)

    trainA    = X_train.copy(); trainA['yA'] = yA_train
    max_n     = trainA['yA'].value_counts().max()
    trainA_bal = pd.concat([
        sk_resample(trainA[trainA['yA'] == c], replace=True,
                    n_samples=max_n, random_state=42)
        for c in [0, 1]
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    XA_bal = trainA_bal.drop(columns=['yA'])
    yA_bal = trainA_bal['yA'].values

    modelA = XGBClassifier(
        objective='binary:logistic',
        n_estimators=n_estimators_a, max_depth=6,
        learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        tree_method='hist', eval_metric='logloss',
        use_label_encoder=False,
    )
    modelA.fit(XA_bal, yA_bal)
    joblib.dump(modelA, _stage_a_path(base_dir))

    stageA_train_acc = float((modelA.predict(XA_bal) == yA_bal).mean())

    # ── 4. Stage B: Baseline / Amusement / Meditation ───────────────────
    mapB    = {0: 0, 2: 1, 3: 2}
    inv_mapB = {0: 0, 1: 2, 2: 3}
    maskB   = (y_train != 1)
    XB_raw  = X_train[maskB].copy()
    yB_raw  = np.array([mapB[v] for v in y_train[maskB]])

    trainB    = XB_raw.copy(); trainB['yB'] = yB_raw
    max_nB    = trainB['yB'].value_counts().max()
    trainB_bal = pd.concat([
        sk_resample(trainB[trainB['yB'] == c], replace=True,
                    n_samples=max_nB, random_state=42)
        for c in sorted(trainB['yB'].unique())
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    XB_bal = trainB_bal.drop(columns=['yB'])
    yB_bal = trainB_bal['yB'].values

    modelB = XGBClassifier(
        objective='multi:softprob', num_class=3,
        n_estimators=n_estimators_b, max_depth=6,
        learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        tree_method='hist', eval_metric='mlogloss',
        use_label_encoder=False,
    )
    modelB.fit(XB_bal, yB_bal)
    joblib.dump(modelB, _stage_b_path(base_dir))

    stageB_train_acc = float((modelB.predict(XB_bal) == yB_bal).mean())

    # ── 5. Evaluate on test set ──────────────────────────────────────────
    y_pred   = _two_stage_predict(modelA, modelB, X_test, thr=stress_threshold)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    acc      = accuracy_score(y_test, y_pred)
    correct_mask = y_test == y_pred

    return {
        'mode':                  'retrain',
        'test_accuracy':         f"{acc*100:.1f}%",
        'macro_f1':              f"{macro_f1*100:.1f}%",
        'total_samples':         len(y_test),
        'correct_predictions':   int(correct_mask.sum()),
        'wrong_predictions':     int((~correct_mask).sum()),
        'total_windows':         len(df),
        'train_windows':         len(train_idx),
        'test_windows':          len(test_idx),
        'n_features':            len(feature_cols),
        'stress_threshold':      stress_threshold,
        'stageA_train_acc':      f"{stageA_train_acc*100:.1f}%",
        'stageB_train_acc':      f"{stageB_train_acc*100:.1f}%",
        'classification_report': classification_report(y_test, y_pred, target_names=CLASS_NAMES),
        'per_class_metrics':     _build_per_class_metrics(y_test, y_pred, CLASS_NAMES),
        'cm_base64':             _plot_confusion_matrix(y_test, y_pred, CLASS_NAMES),
        'class_acc_base64':      _plot_class_accuracy(y_test, y_pred, CLASS_NAMES),
        'dist_base64':           _plot_prediction_distribution(y_pred, CLASS_NAMES),
        'train_acc_base64':      _plot_training_accuracy({
                                     'Stage A\n(Binary)': stageA_train_acc,
                                     'Stage B\n(3-class)': stageB_train_acc,
                                 }),
        'class_names':           CLASS_NAMES,
    }