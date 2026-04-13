
# ─────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────
import os
import uuid
import zipfile
import shutil

# ─────────────────────────────────────────────
# Third-party
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Django
# ─────────────────────────────────────────────
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect

# ─────────────────────────────────────────────
# Local
# ─────────────────────────────────────────────
from users.forms import UserRegistrationForm
from users.models import UserRegistrationModel


# ═══════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════

FS_TARGET   = 4.0
DEFAULT_THR = 0.35

REQUIRED_COLS = [
    "W_EDA", "W_TEMP", "W_BVP", "W_ACCx", "W_ACCy", "W_ACCz",
    "C_ECG", "C_RESP", "C_EMG", "C_EDA", "C_TEMP", "C_ACCx", "C_ACCy", "C_ACCz",
]

# Model paths
_STAGEA_PATH = os.path.join(settings.MEDIA_ROOT, "models", "stageA_stress_vs_nonstress.pkl")
_STAGEB_PATH = os.path.join(settings.MEDIA_ROOT, "models", "stageB_nonstress_3class.pkl")
_COLS_PATH   = os.path.join(settings.MEDIA_ROOT, "models", "feature_columns.pkl")

# Uploads folder  (add  UPLOADS_ROOT = BASE_DIR / 'media' / 'uploads'  to settings.py)
_UPLOADS_ROOT = getattr(settings, "UPLOADS_ROOT",
                         os.path.join(settings.MEDIA_ROOT, "uploads"))


# ═══════════════════════════════════════════════════════
# Lazy model loader  — avoids crash on startup if
# .pkl files haven't been created yet
# ═══════════════════════════════════════════════════════

_cache = {}

def _load_models():
    """Load and cache models on first call."""
    if _cache:
        return _cache["A"], _cache["B"], _cache["cols"]

    import joblib
    for path, label in [(_STAGEA_PATH, "Stage A model"),
                         (_STAGEB_PATH, "Stage B model"),
                         (_COLS_PATH,   "feature_columns.pkl")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found at:\n{path}\n\n"
                "Run Full Retrain from the Training page first, or copy your "
                "saved .pkl files to  MEDIA_ROOT/models/")

    _cache["A"]    = joblib.load(_STAGEA_PATH)
    _cache["B"]    = joblib.load(_STAGEB_PATH)
    _cache["cols"] = joblib.load(_COLS_PATH)
    return _cache["A"], _cache["B"], _cache["cols"]

def _invalidate_model_cache():
    """Call after retraining so next prediction reloads fresh models."""
    _cache.clear()


# ═══════════════════════════════════════════════════════
# Auth helpers
# ═══════════════════════════════════════════════════════

def user_login_required(view_func):
    def wrapper(request, *args, **kwargs):
        if "loggeduser" not in request.session:
            return redirect("please_login")
        return view_func(request, *args, **kwargs)
    return wrapper


# ═══════════════════════════════════════════════════════
# Auth views
# ═══════════════════════════════════════════════════════

def UserLogin(request):
    return render(request, "UserLogin.html")


def UserRegisterForm(request):
    form = UserRegistrationForm()
    return render(request, "UserRegistrationForm.html", {"form": form})


def UserRegisterActions(request):
    if request.method == "POST":
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "You have been successfully registered!")
            return redirect("AdminLogin")
        else:
            messages.error(request, "Email or Mobile already exists")
    else:
        form = UserRegistrationForm()
    return render(request, "UserRegistrationForm.html", {"form": form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid  = request.POST.get("loginid")
        password = request.POST.get("password")

        try:
            user = UserRegistrationModel.objects.get(loginid=loginid, password=password)
            request.session.update({
                "loggeduser":             user.Username,
                "loginid":                user.loginid,
                "email":                  user.email,
                "last_training_time":     None,
                "last_model_accuracy":    None,
                "last_model_name":        None,
                "last_correct_preds":     None,
                "last_incorrect_preds":   None,
                "last_prediction_time":   None,
            })
            return redirect("Userbase")

        except UserRegistrationModel.DoesNotExist:
            messages.error(request, "Invalid Login ID or Password")
            return redirect("UserLogin")

    return redirect("UserLogin")


@user_login_required
def Userbase(request):
    return render(request, "users/userhome.html")


# ═══════════════════════════════════════════════════════
# Training view
# ═══════════════════════════════════════════════════════

_MODEL_STAGES = [
    {"stage": "Stage A", "type": "Feature Extraction",  "info": "14 signals → stats + FFT per 60-second window"},
    {"stage": "Stage A", "type": "Oversampling",         "info": "Balanced classes via random upsampling"},
    {"stage": "Stage A", "type": "XGBClassifier",        "info": "Binary — Stress vs Non-Stress"},
    {"stage": "Stage A", "type": "Threshold",            "info": "Stress probability > 0.35 → positive"},
    {"stage": "Stage B", "type": "Filter",               "info": "Non-stress windows only"},
    {"stage": "Stage B", "type": "Oversampling",         "info": "Balanced 3-class distribution"},
    {"stage": "Stage B", "type": "XGBClassifier",        "info": "3-class — Baseline / Amusement / Meditation"},
    {"stage": "Output",  "type": "Two-Stage Decision",   "info": "Final 4-class prediction merged from both stages"},
]

_SIGNAL_INFO = [
    {"name": "W_EDA",  "source": "Wrist", "fs": "4 Hz",     "fft": False},
    {"name": "W_TEMP", "source": "Wrist", "fs": "4 Hz",     "fft": False},
    {"name": "W_BVP",  "source": "Wrist", "fs": "64→4 Hz",  "fft": True},
    {"name": "W_ACC",  "source": "Wrist", "fs": "32→4 Hz",  "fft": True},
    {"name": "C_ECG",  "source": "Chest", "fs": "700→4 Hz", "fft": True},
    {"name": "C_RESP", "source": "Chest", "fs": "700→4 Hz", "fft": True},
    {"name": "C_EMG",  "source": "Chest", "fs": "700→4 Hz", "fft": False},
    {"name": "C_EDA",  "source": "Chest", "fs": "700→4 Hz", "fft": False},
    {"name": "C_TEMP", "source": "Chest", "fs": "700→4 Hz", "fft": False},
    {"name": "C_ACC",  "source": "Chest", "fs": "700→4 Hz", "fft": True},
]


def Training(request):
    from .utils import quick_evaluate, full_retrain  # lazy import

    context = {
        "model_stages": _MODEL_STAGES,
        "signal_info":  _SIGNAL_INFO,
    }

    if request.method == "POST":
        mode = request.POST.get("mode", "fast")          # 'fast' | 'slow'

        # Hyperparams (slow/retrain only)
        window_sec  = int(request.POST.get("window_sec",     60))
        step_sec    = int(request.POST.get("step_sec",       15))
        n_est_a     = int(request.POST.get("n_estimators_a", 700))
        n_est_b     = int(request.POST.get("n_estimators_b", 800))
        stress_thr  = float(request.POST.get("stress_threshold", 0.35))

        context.update({
            "selected_mode":   mode,
            "form_window_sec": window_sec,
            "form_step_sec":   step_sec,
            "form_n_est_a":    n_est_a,
            "form_n_est_b":    n_est_b,
            "form_stress_thr": stress_thr,
        })

        try:
            if mode == "fast":
                context["training_results"] = quick_evaluate(settings.BASE_DIR)

            else:   # slow = full retrain
                context["training_results"] = full_retrain(
                    settings.BASE_DIR,
                    window_sec=window_sec,
                    step_sec=step_sec,
                    n_estimators_A=n_est_a,     # matches utils.py signature
                    n_estimators_B=n_est_b,
                    val_split=1.0 - stress_thr, # use stress_thr field as val_split proxy
                )
                _invalidate_model_cache()       # force fresh reload on next prediction

        except FileNotFoundError as e:
            context["training_error"] = str(e)
        except ImportError as e:
            context["training_error"] = (
                f"Missing dependency: {e}.\n"
                "Install with:  pip install xgboost scikit-learn joblib pandas numpy seaborn"
            )
        except Exception as e:
            context["training_error"] = f"Operation failed: {e}"

    return render(request, "users/Training.html", context)


# ═══════════════════════════════════════════════════════
# Feature extraction helpers
# ═══════════════════════════════════════════════════════

def _fft_features(x, fs=FS_TARGET):
    x = np.asarray(x, dtype=np.float32) - np.mean(x)
    n = len(x)
    if n < 8:
        return 0.0, 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag   = np.abs(np.fft.rfft(x))
    dom_freq    = float(freqs[int(np.argmax(mag[1:]) + 1)]) if len(mag) > 1 else 0.0
    spec_energy = float(np.sum(mag ** 2) / n)
    return dom_freq, spec_energy


def _extract_stats(x, prefix, use_fft=False):
    x = np.asarray(x, dtype=np.float32)
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    slope = float(np.polyfit(np.arange(len(x), dtype=np.float32), x, 1)[0])
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
        dom, spec = _fft_features(x)
        feats[f"{prefix}_dom_freq"]    = dom
        feats[f"{prefix}_spec_energy"] = spec
    return feats


def _features_from_df(df: pd.DataFrame) -> dict:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    feats = {}
    feats.update(_extract_stats(df["W_EDA"].values,  "W_EDA",  use_fft=False))
    feats.update(_extract_stats(df["W_TEMP"].values, "W_TEMP", use_fft=False))
    feats.update(_extract_stats(df["W_BVP"].values,  "W_BVP",  use_fft=True))
    feats.update(_extract_stats(df["W_ACCx"].values, "W_ACCx", use_fft=True))
    feats.update(_extract_stats(df["W_ACCy"].values, "W_ACCy", use_fft=True))
    feats.update(_extract_stats(df["W_ACCz"].values, "W_ACCz", use_fft=True))
    feats.update(_extract_stats(df["C_ECG"].values,  "C_ECG",  use_fft=True))
    feats.update(_extract_stats(df["C_RESP"].values, "C_RESP", use_fft=True))
    feats.update(_extract_stats(df["C_EMG"].values,  "C_EMG",  use_fft=False))
    feats.update(_extract_stats(df["C_EDA"].values,  "C_EDA",  use_fft=False))
    feats.update(_extract_stats(df["C_TEMP"].values, "C_TEMP", use_fft=False))
    feats.update(_extract_stats(df["C_ACCx"].values, "C_ACCx", use_fft=True))
    feats.update(_extract_stats(df["C_ACCy"].values, "C_ACCy", use_fft=True))
    feats.update(_extract_stats(df["C_ACCz"].values, "C_ACCz", use_fft=True))
    return feats


def _make_feature_row(feature_dict: dict, feat_cols: list) -> pd.DataFrame:
    row = {col: 0.0 for col in feat_cols}
    for k, v in feature_dict.items():
        if k in row:
            row[k] = float(v)
    return pd.DataFrame([row], columns=feat_cols)


# ═══════════════════════════════════════════════════════
# Two-stage prediction
# ═══════════════════════════════════════════════════════

_LABEL_NAMES = {0: "Baseline", 1: "Stress", 2: "Amusement", 3: "Meditation"}
_INV_MAP_B   = {0: 0, 1: 2, 2: 3}

def _predict_two_stage(feature_dict: dict, thr: float = DEFAULT_THR) -> dict:
    modelA, modelB, feat_cols = _load_models()
    X = _make_feature_row(feature_dict, feat_cols)

    p_stress = float(modelA.predict_proba(X)[0][1])
    is_stress = p_stress > thr

    if is_stress:
        final_label = 1
        confidence  = p_stress
        b_probs     = None
    else:
        probs_b     = modelB.predict_proba(X)[0]
        b_pred      = int(np.argmax(probs_b))
        final_label = _INV_MAP_B[b_pred]
        confidence  = float(np.max(probs_b))
        b_probs     = probs_b

    return {
        "label":             _LABEL_NAMES[final_label],
        "confidence":        confidence,
        "p_stress":          p_stress,
        "stageA_threshold":  thr,
        "stageB_probs":      b_probs,
    }


# ═══════════════════════════════════════════════════════
# E4 ZIP helpers
# ═══════════════════════════════════════════════════════

def _read_e4_csv(path: str, n_cols: int, col_names: list) -> tuple:
    raw = pd.read_csv(path, header=None)
    if raw.shape[0] < 3:
        raise ValueError(f"{os.path.basename(path)} is too short.")
    try:
        fs_hz = float(raw.iloc[1, 0])
    except Exception:
        raise ValueError(f"Cannot read sampling rate from {os.path.basename(path)}")

    data = raw.iloc[2:].reset_index(drop=True)
    if n_cols == 1:
        df = pd.DataFrame({col_names[0]: data.iloc[:, 0].astype(float).to_numpy()})
    else:
        if data.shape[1] < n_cols:
            raise ValueError(f"{os.path.basename(path)}: expected {n_cols} columns")
        df = pd.DataFrame(data.iloc[:, :n_cols].astype(float).to_numpy(), columns=col_names)
    return df, fs_hz


def _build_df_from_e4_folder(folder: str) -> pd.DataFrame:
    def p(name): return os.path.join(folder, name)

    needed = {"ACC": p("ACC.csv"), "BVP": p("BVP.csv"),
              "EDA": p("EDA.csv"), "TEMP": p("TEMP.csv")}
    missing = [k for k, v in needed.items() if not os.path.exists(v)]
    if missing:
        raise ValueError(f"Missing E4 files: {missing}. Found: {os.listdir(folder)}")

    acc,  fs_acc  = _read_e4_csv(needed["ACC"],  3, ["W_ACCx", "W_ACCy", "W_ACCz"])
    bvp,  fs_bvp  = _read_e4_csv(needed["BVP"],  1, ["W_BVP"])
    eda,  fs_eda  = _read_e4_csv(needed["EDA"],  1, ["W_EDA"])
    temp, fs_temp = _read_e4_csv(needed["TEMP"], 1, ["W_TEMP"])

    def add_time(df, fs):
        df = df.copy()
        df["t"] = np.arange(len(df), dtype=float) / float(fs)
        return df

    acc  = add_time(acc,  fs_acc)
    bvp  = add_time(bvp,  fs_bvp)
    eda  = add_time(eda,  fs_eda)
    temp = add_time(temp, fs_temp)

    t_end = min(acc["t"].iloc[-1], bvp["t"].iloc[-1],
                eda["t"].iloc[-1], temp["t"].iloc[-1])
    base = pd.DataFrame({"t": np.arange(0, t_end, 1.0 / FS_TARGET)})

    for df in [acc, bvp, eda, temp]:
        base = pd.merge_asof(base.sort_values("t"), df.sort_values("t"),
                             on="t", direction="nearest")

    base = base.drop(columns=["t"]).reset_index(drop=True)

    # Chest signals unavailable from E4 — fill with zeros
    for col in ["C_ECG", "C_RESP", "C_EMG", "C_EDA", "C_TEMP",
                "C_ACCx", "C_ACCy", "C_ACCz"]:
        base[col] = 0.0

    return base[REQUIRED_COLS]


# ═══════════════════════════════════════════════════════
# Prediction views
# ═══════════════════════════════════════════════════════

def prediction_home(request):
    return render(request, "users/prediction_home.html")


def stress_prediction_csv(request):
    if request.method == "POST" and request.FILES.get("csv_file"):
        upload = request.FILES["csv_file"]
        thr    = float(request.POST.get("threshold", DEFAULT_THR))

        os.makedirs(_UPLOADS_ROOT, exist_ok=True)
        fs       = FileSystemStorage(location=_UPLOADS_ROOT)
        filename = fs.save(upload.name, upload)
        filepath = os.path.join(_UPLOADS_ROOT, filename)
        workdir  = None

        try:
            if filename.lower().endswith(".zip"):
                workdir = os.path.join(_UPLOADS_ROOT, f"extracted_{uuid.uuid4().hex}")
                os.makedirs(workdir, exist_ok=True)
                with zipfile.ZipFile(filepath, "r") as z:
                    z.extractall(workdir)

                e4_folder = None
                need = {"ACC.csv", "BVP.csv", "EDA.csv", "TEMP.csv"}
                for root, _, files in os.walk(workdir):
                    if need.issubset(set(files)):
                        e4_folder = root
                        break

                if e4_folder is None:
                    raise ValueError(
                        "Could not find Empatica E4 files inside ZIP.\n"
                        "Expected: ACC.csv, BVP.csv, EDA.csv, TEMP.csv")

                df = _build_df_from_e4_folder(e4_folder)

            else:
                df = pd.read_csv(filepath)

            feats = _features_from_df(df)
            out   = _predict_two_stage(feats, thr)

            return render(request, "users/prediction_result.html", {
                "mode":          "ZIP Upload" if filename.lower().endswith(".zip") else "CSV Upload",
                "filename":      filename,
                "label":         out["label"],
                "confidence":    f"{out['confidence']*100:.2f}%",
                "p_stress":      f"{out['p_stress']*100:.2f}%",
                "threshold":     out["stageA_threshold"],
                "required_cols": ", ".join(REQUIRED_COLS),
            })

        except Exception as e:
            return render(request, "users/prediction_csv.html", {
                "error":             str(e),
                "required_cols":     ", ".join(REQUIRED_COLS),
                "default_threshold": DEFAULT_THR,
            })

        finally:
            if workdir and os.path.exists(workdir):
                shutil.rmtree(workdir, ignore_errors=True)

    return render(request, "users/prediction_csv.html", {
        "required_cols":     ", ".join(REQUIRED_COLS),
        "default_threshold": DEFAULT_THR,
    })


def stress_prediction_manual(request):
    if request.method == "POST":
        thr = float(request.POST.get("threshold", DEFAULT_THR))

        try:
            values = {}
            for col in REQUIRED_COLS:
                v = request.POST.get(col, "").strip()
                if not v:
                    raise ValueError(f"Missing value for {col}")
                values[col] = float(v)

            n   = int(60 * FS_TARGET)   # 240 samples
            t   = np.arange(n, dtype=np.float32) / FS_TARGET
            rng = np.random.default_rng()

            def make_signal(base, noise, amp=0.0, freq=0.2, trend=0.0):
                return (base
                        + trend * (t - t.mean())
                        + amp * np.sin(2 * np.pi * freq * t)
                        + rng.normal(0.0, noise, n)).astype(np.float32)

            data = {
                "W_EDA":  make_signal(values["W_EDA"],  max(0.02, abs(values["W_EDA"])  * 0.06), amp=max(0.01, abs(values["W_EDA"])  * 0.03), freq=0.08),
                "W_TEMP": make_signal(values["W_TEMP"], 0.03,  amp=0.02, freq=0.03, trend=0.0005),
                "W_BVP":  make_signal(values["W_BVP"],  max(0.30, abs(values["W_BVP"])  * 0.03), amp=max(0.80, abs(values["W_BVP"])  * 0.06), freq=1.2),
                "W_ACCx": make_signal(values["W_ACCx"], max(0.05, abs(values["W_ACCx"]) * 0.10), amp=max(0.05, abs(values["W_ACCx"]) * 0.06), freq=0.25),
                "W_ACCy": make_signal(values["W_ACCy"], max(0.05, abs(values["W_ACCy"]) * 0.10), amp=max(0.05, abs(values["W_ACCy"]) * 0.06), freq=0.25),
                "W_ACCz": make_signal(values["W_ACCz"], max(0.05, abs(values["W_ACCz"]) * 0.10), amp=max(0.05, abs(values["W_ACCz"]) * 0.06), freq=0.25),
                "C_ECG":  make_signal(values["C_ECG"],  max(0.02, abs(values["C_ECG"])  * 0.25), amp=max(0.02, abs(values["C_ECG"])  * 0.20), freq=1.2),
                "C_RESP": make_signal(values["C_RESP"], max(0.05, abs(values["C_RESP"]) * 0.05), amp=max(0.20, abs(values["C_RESP"]) * 0.08), freq=0.25),
                "C_EMG":  make_signal(values["C_EMG"],  max(0.01, abs(values["C_EMG"])  * 0.50), amp=max(0.01, abs(values["C_EMG"])  * 0.30), freq=0.35),
                "C_EDA":  make_signal(values["C_EDA"],  max(0.02, abs(values["C_EDA"])  * 0.06), amp=max(0.01, abs(values["C_EDA"])  * 0.03), freq=0.08),
                "C_TEMP": make_signal(values["C_TEMP"], 0.03, amp=0.02, freq=0.03, trend=0.0005),
                "C_ACCx": make_signal(values["C_ACCx"], max(0.03, abs(values["C_ACCx"]) * 0.08), amp=max(0.03, abs(values["C_ACCx"]) * 0.05), freq=0.25),
                "C_ACCy": make_signal(values["C_ACCy"], max(0.03, abs(values["C_ACCy"]) * 0.08), amp=max(0.03, abs(values["C_ACCy"]) * 0.05), freq=0.25),
                "C_ACCz": make_signal(values["C_ACCz"], max(0.03, abs(values["C_ACCz"]) * 0.08), amp=max(0.03, abs(values["C_ACCz"]) * 0.05), freq=0.25),
            }

            df    = pd.DataFrame(data, columns=REQUIRED_COLS)
            df    = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
            feats = _features_from_df(df)
            out   = _predict_two_stage(feats, thr)

            return render(request, "users/prediction_result.html", {
                "mode":          "Manual Input",
                "filename":      "N/A",
                "label":         out["label"],
                "confidence":    f"{out['confidence']*100:.2f}%",
                "p_stress":      f"{out['p_stress']*100:.2f}%",
                "threshold":     out["stageA_threshold"],
                "required_cols": ", ".join(REQUIRED_COLS),
            })

        except Exception as e:
            return render(request, "users/prediction_manual.html", {
                "error":             str(e),
                "default_threshold": DEFAULT_THR,
                "cols":              REQUIRED_COLS,
            })

    return render(request, "users/prediction_manual.html", {
        "default_threshold": DEFAULT_THR,
        "cols":              REQUIRED_COLS,
    })