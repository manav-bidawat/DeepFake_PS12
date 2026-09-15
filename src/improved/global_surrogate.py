from __future__ import annotations

import argparse
import time
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from tqdm import tqdm

from src.improved.explainability_utils import (
    clean_output_dir,
    ensure_dir,
    get_score_column,
    get_top_n_files,
    load_audio,
    log_error,
    write_manifest,
    write_summary,
)


def _select_utt_ids(scores_csv: str | Path, top_n: int) -> list[str]:
    if top_n and top_n > 0:
        return get_top_n_files(scores_csv, top_n)
    df = pd.read_csv(scores_csv)
    return df["utt_id"].astype(str).tolist()


def _load_scores(scores_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)
    if "utt_id" not in df.columns:
        raise ValueError("scores_csv must contain utt_id")
    return df


def _resolve_audio_path(input_dir: Path, utt_id: str) -> Path | None:
    patterns = [f"**/*{utt_id}*.wav", f"**/*{utt_id}*.flac", f"**/{utt_id}.wav", f"**/{utt_id}.flac"]
    for pattern in patterns:
        matches = sorted(input_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _safe_stats(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0, 0.0
    return float(np.nanmean(values)), float(np.nanstd(values))


def _band_energy(waveform: np.ndarray, sr: int, low: float, high: float) -> float:
    nyq = sr / 2.0
    if low <= 0 and high >= nyq:
        filtered = waveform
    elif low <= 0:
        sos = butter(6, high / nyq, btype="lowpass", output="sos")
        filtered = sosfiltfilt(sos, waveform)
    elif high >= nyq:
        sos = butter(6, low / nyq, btype="highpass", output="sos")
        filtered = sosfiltfilt(sos, waveform)
    else:
        sos = butter(6, [low / nyq, high / nyq], btype="bandpass", output="sos")
        filtered = sosfiltfilt(sos, waveform)
    return float(np.mean(filtered**2))


def extract_features(waveform: np.ndarray, sr: int) -> tuple[np.ndarray, list[str]]:
    waveform = np.asarray(waveform, dtype=np.float32)
    eps = 1e-8
    feats = []
    names: list[str] = []

    def add_stat_block(prefix: str, matrix: np.ndarray):
        nonlocal feats, names
        for idx in range(matrix.shape[0]):
            mean_v, std_v = _safe_stats(matrix[idx])
            feats.extend([mean_v, std_v])
            names.extend([f"{prefix}_{idx+1}_mean", f"{prefix}_{idx+1}_std"])

    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13, n_fft=512, hop_length=160)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    add_stat_block("mfcc", mfcc)
    add_stat_block("delta_mfcc", delta)
    add_stat_block("delta2_mfcc", delta2)

    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr, n_fft=512, hop_length=160)
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr, n_fft=512, hop_length=160, roll_percent=0.85)
    zcr = librosa.feature.zero_crossing_rate(y=waveform, frame_length=512, hop_length=160)
    rms = librosa.feature.rms(y=waveform, frame_length=512, hop_length=160)
    spec = np.abs(librosa.stft(waveform, n_fft=512, hop_length=160))
    flux = np.zeros(spec.shape[1], dtype=np.float32)
    if spec.shape[1] > 1:
        flux[1:] = np.mean(np.abs(np.diff(spec, axis=1)), axis=0)
    yin = librosa.yin(waveform, fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=160)

    for prefix, matrix in (
        ("spectral_centroid", centroid),
        ("spectral_rolloff", rolloff),
        ("zcr", zcr),
        ("rms", rms),
    ):
        mean_v, std_v = _safe_stats(matrix)
        feats.extend([mean_v, std_v])
        names.extend([f"{prefix}_mean", f"{prefix}_std"])

    mean_v, std_v = _safe_stats(flux)
    feats.extend([mean_v, std_v])
    names.extend(["spectral_flux_mean", "spectral_flux_std"])

    voiced = yin[~np.isnan(yin)]
    mean_f0, std_f0 = _safe_stats(voiced)
    frac_unvoiced = float(np.mean(np.isnan(yin))) if yin.size else 1.0
    feats.extend([mean_f0, std_f0, frac_unvoiced])
    names.extend(["f0_mean", "f0_std", "f0_frac_unvoiced"])

    bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000)]
    for low, high in bands:
        energy = _band_energy(waveform, sr, low, high)
        feats.append(energy)
        names.append(f"band_{low}_{high}_energy")

    return np.asarray(feats, dtype=np.float32), names


def _fit_tree(X: np.ndarray, y: np.ndarray, depths: list[int], folds: int, seed: int) -> tuple[DecisionTreeClassifier, int, dict[int, float]]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: dict[int, float] = {}
    best_depth = depths[0]
    best_score = -1.0
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=seed)
        cv_score = float(cross_val_score(clf, X, y, cv=cv, scoring="f1").mean())
        scores[depth] = cv_score
        if cv_score > best_score:
            best_score = cv_score
            best_depth = depth
    best_model = DecisionTreeClassifier(max_depth=best_depth, random_state=seed)
    best_model.fit(X, y)
    return best_model, best_depth, scores


def _tree_feature_ranking(model: DecisionTreeClassifier, feature_names: list[str]) -> list[tuple[str, float]]:
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    return [(feature_names[i], float(importances[i])) for i in order if importances[i] > 0][:20]


def _write_feature_plot(output_path: Path, ranking: list[tuple[str, float]]) -> None:
    fig = plt.figure(figsize=(12, 8))
    try:
        top = ranking[:20]
        labels = [x[0] for x in top][::-1]
        values = [x[1] for x in top][::-1]
        plt.barh(labels, values, color="#3366cc")
        plt.xlabel("Feature importance")
        plt.title("Top 20 decision tree features")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Global surrogate explainability")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--aasist-config", type=Path, required=True)
    parser.add_argument("--aasist-checkpoint", type=Path, required=True)
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("explainability/global_surrogate"))
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    start = time.time()
    ensure_dir(args.output_dir)
    clean_output_dir(args.output_dir)
    scores_df = _load_scores(args.scores_csv)
    score_col = get_score_column(args.scores_csv)
    utt_ids = _select_utt_ids(args.scores_csv, args.top_n)

    rows = []
    feature_names: list[str] = []
    files_processed = 0
    files_failed = 0
    files_skipped = 0
    score_accum: list[float] = []

    for utt in tqdm(utt_ids, desc="global_surrogate_features"):
        audio_path = _resolve_audio_path(args.input_dir, utt)
        if audio_path is None:
            files_skipped += 1
            log_error("global_surrogate", f"missing audio for {utt}")
            continue
        try:
            waveform, sr = load_audio(audio_path)
            feats, names = extract_features(waveform.numpy(), sr)
            if not feature_names:
                feature_names = names
            row = {"utt_id": utt, **{k: float(v) for k, v in zip(feature_names, feats)}}
            meta = scores_df[scores_df["utt_id"].astype(str) == str(utt)]
            if meta.empty:
                raise ValueError(f"utt_id missing from scores csv: {utt}")
            aasist_score = float(meta.iloc[0][score_col])
            row["aasist_score"] = aasist_score
            row["aasist_label"] = int(aasist_score >= 0.5)
            rows.append(row)
            files_processed += 1
            score_accum.append(aasist_score)
        except Exception as exc:
            files_failed += 1
            log_error("global_surrogate", f"{utt}: {exc}")

    if not rows:
        raise RuntimeError("No feature rows were extracted")

    feature_df = pd.DataFrame(rows).set_index("utt_id")
    feature_csv = args.output_dir / "feature_matrix.csv"
    feature_df.to_csv(feature_csv)

    X = feature_df[feature_names].to_numpy(dtype=np.float32)
    y = feature_df["aasist_label"].to_numpy(dtype=int)
    raw_scores = feature_df["aasist_score"].to_numpy(dtype=np.float32)

    class_counts = dict(zip(*np.unique(y, return_counts=True)))
    if len(class_counts) < 2:
        raise RuntimeError(f"Surrogate requires at least 2 classes after binarizing AASIST scores at 0.5. Got: {class_counts}")

    tree, best_depth, depth_scores = _fit_tree(X, y, [3, 5, 7], args.cv_folds, seed=1337)
    tree_pred = tree.predict(X)
    tree_accuracy = float((tree_pred == y).mean())
    tree_f1 = float(f1_score(y, tree_pred))
    tree_agreement = float((tree_pred == y).mean())

    rules = export_text(tree, feature_names=feature_names)
    (args.output_dir / "decision_tree_rules.txt").write_text(rules, encoding="utf-8")

    fig = plt.figure(figsize=(24, 12))
    try:
        plot_tree(tree, feature_names=feature_names, class_names=["BONAFIDE", "FAKE"], filled=True, rounded=True, max_depth=3)
        fig.tight_layout()
        fig.savefig(args.output_dir / "decision_tree_plot.png", dpi=180)
    finally:
        plt.close(fig)

    ranking = _tree_feature_ranking(tree, feature_names)
    _write_feature_plot(args.output_dir / "feature_importance_plot.png", ranking)

    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=3000, random_state=1337)
    lr_pipe = Pipeline([("scaler", scaler), ("lr", lr)])
    lr_pipe.fit(X, y)
    lr_pred = lr_pipe.predict(X)
    lr_accuracy = float((lr_pred == y).mean())
    lr_f1 = float(f1_score(y, lr_pred))
    lr_agreement = float((lr_pred == y).mean())

    scaler.fit(X)
    lr.fit(scaler.transform(X), y)
    coef_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "coefficient": lr.coef_[0],
        }
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(args.output_dir / "logistic_regression_coefficients.csv", index=False)

    report_lines = [
        f"Decision tree best depth: {best_depth}",
        f"Decision tree CV F1 scores: {depth_scores}",
        f"Class counts: {class_counts}",
        f"Decision tree fidelity accuracy: {tree_accuracy:.4f}",
        f"Decision tree fidelity F1: {tree_f1:.4f}",
        f"Decision tree agreement rate: {tree_agreement:.4f}",
        f"Logistic regression fidelity accuracy: {lr_accuracy:.4f}",
        f"Logistic regression fidelity F1: {lr_f1:.4f}",
        f"Logistic regression agreement rate: {lr_agreement:.4f}",
        "Top 10 decision tree features:",
    ]
    for feature_name, importance in ranking[:10]:
        report_lines.append(f"  {feature_name}: {importance:.6f}")
    (args.output_dir / "fidelity_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    write_manifest(args.output_dir, [
        args.output_dir / "decision_tree_rules.txt",
        args.output_dir / "decision_tree_plot.png",
        args.output_dir / "logistic_regression_coefficients.csv",
        args.output_dir / "feature_importance_plot.png",
        args.output_dir / "fidelity_report.txt",
        feature_csv,
    ])
    write_summary(
        "global_surrogate",
        args.output_dir,
        files_processed,
        files_failed,
        files_skipped,
        float(np.mean(score_accum)) if score_accum else 0.0,
        time.time() - start,
    )


if __name__ == "__main__":
    main()
