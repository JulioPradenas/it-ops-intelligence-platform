"""Entrena todos los modelos, los serializa y genera dashboard_data.parquet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from itops.config import MODELS_DIR, PROCESSED_DIR, RAW_TICKETS_CSV
from itops.data.features import FEATURE_COLS, build_hourly_features
from itops.models.anomaly import AutoencoderDetector, IsolationForestDetector
from itops.models.escalation import EscalationModel


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(str(Path(__file__).resolve().parents[1] / "mlruns"))
    mlflow.set_experiment("it-ops-intelligence")

    print("Cargando datos...")
    df = pd.read_csv(RAW_TICKETS_CSV, parse_dates=["created_at"])
    print(f"  {len(df):,} tickets cargados")

    with mlflow.start_run(run_name="train_all"):
        mlflow.log_param("n_tickets", len(df))
        mlflow.log_param("escalated_pct", round(df["escalated"].mean(), 4))

        # --- Escalation model ---
        print("Entrenando EscalationModel...")
        escalation_model = EscalationModel(seed=42)
        escalation_model.fit(df)
        escalation_model.save(MODELS_DIR / "escalation_model.pkl")
        print(f"  AUC-ROC:  {escalation_model.eval_metrics_['auc_roc']:.4f}")
        print(f"  PR-AUC:   {escalation_model.eval_metrics_['pr_auc']:.4f}")
        print(f"  Threshold: {escalation_model.threshold_:.3f}")

        mlflow.log_metrics({
            "auc_roc": escalation_model.eval_metrics_["auc_roc"],
            "pr_auc": escalation_model.eval_metrics_["pr_auc"],
            "f1": escalation_model.eval_metrics_["f1"],
            "precision": escalation_model.eval_metrics_["precision"],
            "recall": escalation_model.eval_metrics_["recall"],
            "threshold": escalation_model.threshold_,
        })
        mlflow.log_param("escalation_seed", 42)
        mlflow.log_artifact(str(MODELS_DIR / "escalation_model.pkl"), "models")

        # --- Anomaly detectors ---
        print("Construyendo features horarias...")
        hourly_feat = build_hourly_features(df)
        X = hourly_feat[FEATURE_COLS].values
        mlflow.log_param("n_hourly_windows", len(X))

        print("Entrenando IsolationForestDetector...")
        if_detector = IsolationForestDetector(seed=42)
        if_detector.fit(X)
        if_detector.save(MODELS_DIR / "if_detector.pkl")
        mlflow.log_artifact(str(MODELS_DIR / "if_detector.pkl"), "models")

        print("Entrenando AutoencoderDetector...")
        ae_detector = AutoencoderDetector(seed=42)
        ae_detector.fit(X)
        ae_detector.save(MODELS_DIR / "ae_detector.pkl", MODELS_DIR / "ae_weights.pt")
        mlflow.log_artifact(str(MODELS_DIR / "ae_detector.pkl"), "models")
        mlflow.log_artifact(str(MODELS_DIR / "ae_weights.pt"), "models")

        # --- Pre-compute predictions ---
        print("Pre-computando predicciones de escalación...")
        df_sorted = df.sort_values("created_at").reset_index(drop=True)
        risk_scores = escalation_model.predict_proba(df_sorted)
        predicted_escalation = escalation_model.predict(df_sorted)

        print("Pre-computando anomalías por ventana...")
        hourly_all = build_hourly_features(df_sorted)
        X_all = hourly_all[FEATURE_COLS].values
        if_scores = if_detector.score(X_all)
        ae_scores = ae_detector.score(X_all)
        is_anomaly = if_detector.predict(X_all, percentile=97.0)

        hourly_all["if_score"] = if_scores
        hourly_all["ae_score"] = ae_scores
        hourly_all["is_anomaly"] = is_anomaly

        dashboard_df = df_sorted[
            ["ticket_id", "created_at", "category", "customer_tier", "escalated",
             "response_time_minutes", "priority_initial", "assigned_team"]
        ].copy()
        dashboard_df["risk_score"] = risk_scores
        dashboard_df["predicted_escalation"] = predicted_escalation.astype(bool)
        dashboard_df["_date"] = dashboard_df["created_at"].dt.date
        dashboard_df["_hour"] = dashboard_df["created_at"].dt.hour

        window_data = hourly_all[
            ["date", "hour", "category", "if_score", "ae_score", "is_anomaly"]
        ].rename(columns={"date": "_date", "hour": "_hour"})

        merged = dashboard_df.merge(
            window_data, on=["_date", "_hour", "category"], how="left"
        ).drop(columns=["_date", "_hour"])

        parquet_path = PROCESSED_DIR / "dashboard_data.parquet"
        merged.to_parquet(parquet_path, index=False)
        print(f"  Parquet guardado: {parquet_path} ({len(merged):,} filas, {merged.shape[1]} columnas)")

        metrics = {**escalation_model.eval_metrics_, "threshold": escalation_model.threshold_}
        metrics_path = PROCESSED_DIR / "model_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        mlflow.log_artifact(str(metrics_path))
        print(f"  Métricas guardadas: {metrics_path}")

        anomaly_pct = float(is_anomaly.mean())
        mlflow.log_metric("anomaly_pct_windows", round(anomaly_pct, 4))

    print("\nDone. Run `mlflow ui` to view experiment tracking.")


if __name__ == "__main__":
    main()
