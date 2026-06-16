"""Dashboard Streamlit con 4 vistas sobre datos ITSM pre-computados."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from itops.config import PROCESSED_DIR, RAW_TICKETS_CSV

PARQUET_PATH = PROCESSED_DIR / "dashboard_data.parquet"
METRICS_PATH = PROCESSED_DIR / "model_metrics.json"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet(PARQUET_PATH)


@st.cache_data
def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


@st.cache_data
def load_raw_tickets() -> pd.DataFrame:
    return pd.read_csv(RAW_TICKETS_CSV, parse_dates=["created_at"])


def view_operaciones(df: pd.DataFrame) -> None:
    st.header("Operaciones")

    col1, col2, col3, col4 = st.columns(4)
    pct_escalated = df["escalated"].mean() * 100
    pct_anomaly = df["is_anomaly"].fillna(False).mean() * 100
    mttr = df["response_time_minutes"].mean()
    col1.metric("Total tickets", f"{len(df):,}")
    col2.metric("% Escalados", f"{pct_escalated:.1f}%")
    col3.metric("% Anomalías", f"{pct_anomaly:.1f}%")
    col4.metric("MTTR medio (min)", f"{mttr:.0f}")

    st.subheader("Heatmap de anomalías por hora y categoría")
    df_heat = df.copy()
    df_heat["hour"] = pd.to_datetime(df_heat["created_at"]).dt.hour
    heat_data = df_heat.pivot_table(
        index="hour", columns="category",
        values="is_anomaly", aggfunc="mean", fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heat_data, ax=ax, cmap="YlOrRd", annot=True, fmt=".1%")
    ax.set_title("Tasa de anomalías por hora y categoría")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Top 10 tickets por riesgo")
    top10 = (
        df.nlargest(10, "risk_score")[
            ["ticket_id", "category", "customer_tier", "risk_score", "escalated"]
        ].rename(columns={
            "ticket_id": "Ticket", "category": "Categoría", "customer_tier": "Tier",
            "risk_score": "Riesgo", "escalated": "Escalado",
        })
    )
    st.dataframe(top10, use_container_width=True)

    st.subheader("Narrativa LLM — Explicación por ticket")
    st.caption("Requiere que la API esté corriendo. Selecciona un ticket de alto riesgo y genera la explicación en lenguaje natural.")

    api_url = st.text_input("URL de la API", value="http://localhost:8000", key="api_url")
    top_ids = df.nlargest(20, "risk_score")["ticket_id"].tolist()
    selected_id = st.selectbox("Ticket (top 20 por riesgo)", top_ids)

    if st.button("Generar narrativa con Claude"):
        raw = load_raw_tickets()
        matches = raw[raw["ticket_id"] == selected_id]
        if matches.empty:
            st.error(f"Ticket {selected_id} no encontrado en el dataset raw.")
        else:
            row = matches.iloc[0]
            payload = {
                "ticket": {
                    "ticket_id": str(row["ticket_id"]),
                    "created_at": row["created_at"].isoformat(),
                    "category": str(row["category"]),
                    "subcategory": str(row.get("subcategory", "unknown")),
                    "priority_initial": str(row["priority_initial"]),
                    "customer_tier": str(row["customer_tier"]),
                    "description": str(row["description"]),
                    "response_time_minutes": int(row["response_time_minutes"]),
                    "num_comments": int(row["num_comments"]),
                    "num_reassignments": int(row["num_reassignments"]),
                    "business_hours": bool(row["business_hours"]),
                    "assigned_team": str(row["assigned_team"]),
                }
            }
            with st.spinner("Llamando a la API..."):
                try:
                    import httpx
                    resp = httpx.post(f"{api_url}/explain", json=payload, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        narrative = data["narrative"]
                        st.markdown(f"**Resumen:** {narrative['summary']}")
                        st.markdown(f"**Recomendación:** {narrative['recommendation']}")
                        c1, c2 = st.columns(2)
                        c1.metric("Confianza", f"{narrative['confidence']:.0%}")
                        c2.metric("Proveedor LLM", narrative["provider"])
                        features_str = " · ".join(f["feature"] for f in data["top_features"])
                        st.caption(f"Top SHAP features: {features_str}")
                    else:
                        st.error(f"API respondió {resp.status_code}: {resp.text[:200]}")
                except Exception as exc:
                    st.warning(
                        f"No se pudo conectar a la API en `{api_url}`.\n\n"
                        f"Inicia el servidor con:\n```\n"
                        f"KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 "
                        f"uv run uvicorn itops.api.main:app\n```\n\nError: {exc}"
                    )


def view_compliance(df: pd.DataFrame) -> None:
    st.header("Compliance")

    st.subheader("Tickets predichos como escalación")
    pred_esc = df[df["predicted_escalation"]][
        ["ticket_id", "category", "customer_tier", "risk_score", "priority_initial", "assigned_team"]
    ]
    st.dataframe(pred_esc, use_container_width=True)

    st.subheader("Tendencia mensual de escalaciones por tier")
    df_monthly = df.copy()
    df_monthly["month"] = pd.to_datetime(df_monthly["created_at"]).dt.to_period("M").astype(str)
    trend = (
        df_monthly[df_monthly["escalated"]]
        .groupby(["month", "customer_tier"])
        .size()
        .unstack(fill_value=0)
    )
    st.line_chart(trend)

    st.subheader("Tiempo de respuesta por prioridad")
    fig, ax = plt.subplots(figsize=(8, 4))
    priorities = sorted(df["priority_initial"].dropna().unique())
    ax.boxplot(
        [df[df["priority_initial"] == p]["response_time_minutes"].dropna() for p in priorities],
        tick_labels=priorities,
    )
    ax.set_xlabel("Prioridad")
    ax.set_ylabel("Tiempo de respuesta (min)")
    ax.set_title("Distribución tiempo de respuesta por prioridad")
    st.pyplot(fig)
    plt.close(fig)


def view_estrategica(df: pd.DataFrame, metrics: dict) -> None:
    st.header("Estratégica")

    col1, col2, col3 = st.columns(3)
    col1.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.3f}")
    col2.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}")
    col3.metric("Threshold óptimo", f"{metrics.get('threshold', 0):.2f}")

    st.subheader("Evolución % escalación semanal")
    df_weekly = df.copy()
    df_weekly["week"] = pd.to_datetime(df_weekly["created_at"]).dt.to_period("W").astype(str)
    weekly_rate = df_weekly.groupby("week")["escalated"].mean() * 100
    st.area_chart(weekly_rate.rename("% escalación"))

    st.subheader("Costo estimado evitado")
    tp = int((df["escalated"] & df["predicted_escalation"]).sum())
    st.metric("Verdaderos positivos detectados", f"{tp:,}")
    st.metric("Costo estimado evitado (USD)", f"${tp * 500:,.0f}")
    st.caption("Asunción: cada escalación detectada a tiempo evita $500 en costo operacional.")


def view_como_lo_hice() -> None:
    st.header("Cómo lo hice")
    st.caption(
        "Esta vista está diseñada para que recruiters y colegas entiendan el proyecto sin leer código."
    )

    with st.expander("Problema de negocio"):
        st.write(
            "Las operaciones IT reciben miles de tickets diariamente. Sin visibilidad temprana "
            "de qué tickets van a escalar, los equipos reaccionan en lugar de anticipar — "
            "aumentando tiempos de resolución, costos y frustración del cliente. "
            "Este proyecto construye un sistema de inteligencia operacional que detecta anomalías "
            "en el volumen de incidentes, predice qué tickets escalarán y genera explicaciones "
            "en lenguaje natural para los equipos de soporte."
        )

    with st.expander("Arquitectura"):
        st.markdown(
            "**Fase 1-2 — Datos y anomalías:** 50k tickets sintéticos con patrones realistas. "
            "Isolation Forest (baseline) y Autoencoder MLP en PyTorch para ventanas horarias.\n\n"
            "**Fase 3 — Predicción de escalación:** LightGBM con split temporal 80/20 y threshold "
            "optimizado por costo asimétrico (FN:FP = 5:1). SHAP TreeExplainer por ticket.\n\n"
            "**Fase 4 — Narrativas LLM:** Claude Haiku genera resúmenes en español con fallback a "
            "flan-t5-small para entornos offline. SQLite para caché de deduplicación.\n\n"
            "**Fase 5 — API y Dashboard:** FastAPI expone los modelos. Streamlit consume parquet "
            "pre-computado — sin dependencia de la API para demos."
        )

    with st.expander("Stack técnico"):
        st.table(pd.DataFrame({
            "Capa": [
                "Datos", "ML Anomalías", "ML Escalación",
                "Explicabilidad", "LLM", "API", "Dashboard",
            ],
            "Herramienta": [
                "pandas, Faker",
                "scikit-learn (Isolation Forest), PyTorch (Autoencoder MLP)",
                "LightGBM",
                "SHAP TreeExplainer",
                "Claude Haiku (primario), flan-t5-small (fallback offline)",
                "FastAPI, uvicorn, Pydantic v2",
                "Streamlit, seaborn, matplotlib",
            ],
        }))

    with st.expander("Decisiones clave"):
        st.markdown(
            "- **LightGBM sobre XGBoost** — soporte nativo de categóricas; `n_jobs=1` evita "
            "el conflicto OpenMP con PyTorch en macOS.\n"
            "- **Threshold por costo asimétrico** — FN:FP = 5:1. Un ticket que escala sin "
            "detectarse cuesta 5× más que una falsa alarma.\n"
            "- **Claude + fallback HF** — calidad de narrativas con Haiku; flan-t5-small "
            "permite ejecutar en CI/offline sin API key.\n"
            "- **Dashboard independiente de la API** — el parquet pre-computado permite demos "
            "sin levantar el servidor; más robusto para presentaciones."
        )

    with st.expander("Métricas del modelo"):
        metrics = load_metrics()
        if metrics:
            col1, col2 = st.columns(2)
            col1.metric("AUC-ROC", f"{metrics.get('auc_roc', 'N/A'):.3f}")
            col2.metric("PR-AUC", f"{metrics.get('pr_auc', 'N/A'):.3f}")
            col1.metric("F1 Score", f"{metrics.get('f1', 'N/A'):.3f}")
            col2.metric("Threshold óptimo", f"{metrics.get('threshold', 'N/A'):.2f}")
        else:
            st.info("Ejecuta `scripts/train_all.py` para generar las métricas.")

    with st.expander("Código fuente"):
        st.markdown(
            "Repositorio completo: "
            "[JulioPradenas/it-ops-intelligence-platform]"
            "(https://github.com/JulioPradenas/it-ops-intelligence-platform)"
        )


def main() -> None:
    st.set_page_config(page_title="IT Ops Intelligence", layout="wide")
    st.title("IT Operations Intelligence Platform")

    vista = st.sidebar.radio(
        "Vista", ["Operaciones", "Compliance", "Estratégica", "Cómo lo hice"]
    )

    if vista == "Cómo lo hice":
        view_como_lo_hice()
        return

    df = load_data()
    metrics = load_metrics()

    if vista == "Operaciones":
        view_operaciones(df)
    elif vista == "Compliance":
        view_compliance(df)
    elif vista == "Estratégica":
        view_estrategica(df, metrics)


if __name__ == "__main__":
    main()
