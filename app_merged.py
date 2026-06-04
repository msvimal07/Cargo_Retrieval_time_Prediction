import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cargo Retrieval Time Prediction",
    page_icon="🏭",
    layout="wide"
)

# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).parent

# ============================================================
# LOAD MODEL & DATA
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "cargo_retrieval_best_model.pkl")
    features = joblib.load(BASE_DIR / "cargo_retrieval_features.pkl")
    return model, features


@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\VimalM\Downloads\warehouse_retrieval_50k_updated.csv")
    return df


model, FEATURES = load_model()
df_raw = load_data()

# ============================================================
# COLORS
# ============================================================

BLUE   = "#2563EB"
TEAL   = "#0D9488"
AMBER  = "#D97706"
CORAL  = "#E85D24"
PURPLE = "#7C3AED"
GREEN  = "#16A34A"
GRAY   = "#6B7280"

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 1rem;
}

.stButton > button {
    background-color: #1565ff;
    color: white;
    border-radius: 10px;
    height: 52px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.result-box {
    background-color: #dbeafe;
    padding: 28px;
    border-radius: 12px;
    font-size: 24px;
    font-weight: bold;
    color: #0b4aa2;
    text-align: center;
    margin-top: 16px;
}

.result-box-high {
    background-color: #fee2e2;
    padding: 28px;
    border-radius: 12px;
    font-size: 24px;
    font-weight: bold;
    color: #991b1b;
    text-align: center;
    margin-top: 16px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<h1 style='font-size:38px; font-weight:700;'>
🏭 Warehouse Cargo Retrieval Time Prediction
</h1>
<p style='color:gray;'>
Predict cargo retrieval time using ML
</p>
""", unsafe_allow_html=True)

st.markdown("---")


# TABS
tab1, tab2 = st.tabs(["🔮 Prediction","📊 EDA Dashboard"])

with tab1:
    # ============================================================
    # INPUT SECTION
    # ============================================================

    col1, col2, col3 = st.columns(3)

    # ============================================================
    # COLUMN 1: Location
    # ============================================================

    with col1:
        st.subheader("Location")

        warehouse_zone = st.selectbox(
            "Warehouse Zone",
            ["Zone_A", "Zone_B", "Zone_C", "Zone_D"]
        )

        num_aisles = st.number_input(
            "Number of Aisles to Visit",
            min_value=1,
            max_value=10,
            value=1,
            step=1
        )

        aisle_numbers = []
        for i in range(int(num_aisles)):
            aisle_val = st.number_input(
                f"Aisle Number {i + 1}",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                key=f"aisle_number_{i + 1}"
            )
            aisle_numbers.append(int(aisle_val))

        # Model expects one aisle_number feature; use mean aisle as representative.
        aisle_number = int(round(sum(aisle_numbers) / len(aisle_numbers)))

        shelf_level = st.selectbox(
            "Shelf Level",
            ["Floor", "Mid", "High"]
        )

        distance_from_dock = st.slider(
            "Distance From Dock (meters)",
            5,
            500,
            150
        )

        storage_type = st.selectbox(
            "Storage Type",
            ["pallet_rack", "bulk_floor", "cold_storage", "hazmat"]
        )

    # ============================================================
    # COLUMN 2: Congestion & Timing
    # ============================================================

    with col2:
        st.subheader("Congestion & Timing")

        active_retrievals = st.slider(
            "Active Retrievals",
            0,
            40,
            10
        )

        dock_queue = st.slider(
            "Dock Queue Length",
            0,
            25,
            5
        )

        aisle_traffic = st.selectbox(
            "Aisle Traffic Level",
            ["Low", "Medium", "High"]
        )

        # Date/Time inputs (used for feature engineering only)
        request_date = st.date_input("Request Date")
        request_time = st.time_input("Request Time")

    # ============================================================
    # COLUMN 3: Shipment & Operator
    # ============================================================

    with col3:
        st.subheader("Shipment & Operator")

        num_packages = st.number_input(
            "Number of Packages",
            1,
            200,
            20
        )

        total_weight = st.number_input(
            "Total Weight (kg)",
            1.0,
            5000.0,
            300.0
        )

        requires_forklift = st.selectbox(
            "Requires Forklift?",
            ["No", "Yes"]
        )

        fragile_handling = st.selectbox(
            "Fragile Handling?",
            ["No", "Yes"]
        )

        special_equipment = st.selectbox(
            "Special Equipment",
            ["none", "cold_gear", "hazmat_suit"]
        )

        operator_experience = st.slider(
            "Operator Experience (years)",
            0,
            25,
            5
        )

        operator_fatigue = st.slider(
            "Operator Shift Fatigue",
            0.0,
            12.0,
            4.0
        )

    # ============================================================
    # PREDICTION BUTTON
    # ============================================================

    st.markdown("---")

    if st.button("🔮 Predict Retrieval Time"):

        # ========================================================
        # STEP 1: Extract datetime components (for feature engineering)
        # ========================================================

        request_datetime = datetime.combine(request_date, request_time)
        request_hour = request_datetime.hour
        request_dayofweek = request_datetime.weekday()  # 0=Monday, 6=Sunday
        request_month = request_datetime.month

        # ========================================================
        # STEP 2: Feature Engineering (EXACTLY as in training)
        # ========================================================

        # Busy aisles set
        busy_aisles = {5, 10, 15, 20, 25, 30}
        is_busy_aisle = int(any(a in busy_aisles for a in aisle_numbers))

        # Peak hour flag (8-11 AM, 4-7 PM)
        is_peak_hour = int((8 <= request_hour <= 11) or (16 <= request_hour <= 19))

        # Day flags
        is_monday = int(request_dayofweek == 0)
        is_weekend = int(request_dayofweek >= 5)

        # Binary conversions
        forklift_int = 1.0 if requires_forklift == "Yes" else 0.0
        fragile_int = 1.0 if fragile_handling == "Yes" else 0.0

        # Traffic mapping
        traffic_map = {"Low": 0, "Medium": 1, "High": 2}
    
        # Congestion score
        congestion_score = (
            active_retrievals * 0.5 +
            dock_queue * 0.3 +
            traffic_map[aisle_traffic] * 5
        )

        # Heavy cargo without forklift flag
        heavy_no_forklift = int(total_weight > 600 and forklift_int == 0)

        # Complexity score
        complexity_score = (
            num_aisles * 2 +
            num_packages * 0.05 +
            forklift_int * 3 +
            fragile_int * 2
        )

        # Log weight (reduces skewness)
        log_weight = np.log1p(total_weight)

        # Ordinal encodings
        shelf_enc = {"Floor": 0, "Mid": 1, "High": 2}[shelf_level]
        traffic_enc = {"Low": 0, "Medium": 1, "High": 2}[aisle_traffic]
        zone_enc = {"Zone_A": 0, "Zone_B": 1, "Zone_C": 2, "Zone_D": 3}[warehouse_zone]

        # ========================================================
        # STEP 3: Build input dictionary with EXACT training features
        # ========================================================

        input_dict = {
            # === Core numerical features ===
            "aisle_number": aisle_number,
            "distance_from_dock_meters": distance_from_dock,
            "active_retrievals_count": active_retrievals,
            "dock_queue_length": float(dock_queue),
            "shift_hour": request_hour,  # Use request_hour as shift_hour
            "num_packages": num_packages,
            "requires_forklift": forklift_int,
            "fragile_handling": fragile_int,
            "num_aisles_to_visit": num_aisles,
            "operator_experience_years": operator_experience,
            "operator_shift_fatigue": operator_fatigue,
        
            # === Engineered features ===
            "is_peak_hour": is_peak_hour,
            "is_monday": is_monday,
            "is_weekend": is_weekend,
            "is_busy_aisle": is_busy_aisle,
            "congestion_score": congestion_score,
            "heavy_no_forklift": heavy_no_forklift,
            "complexity_score": complexity_score,
            "log_weight": log_weight,
        
            # === Ordinal encoded features ===
            "shelf_level_enc": shelf_enc,
            "aisle_traffic_enc": traffic_enc,
            "warehouse_zone_enc": zone_enc,
        
            # === One-hot: Storage Type (4 categories) ===
            "storage_type_bulk_floor": 1 if storage_type == "bulk_floor" else 0,
            "storage_type_cold_storage": 1 if storage_type == "cold_storage" else 0,
            "storage_type_hazmat": 1 if storage_type == "hazmat" else 0,
            "storage_type_pallet_rack": 1 if storage_type == "pallet_rack" else 0,
        
            # === One-hot: Special Equipment (3 categories) ===
            "special_equipment_needed_cold_gear": 1 if special_equipment == "cold_gear" else 0,
            "special_equipment_needed_hazmat_suit": 1 if special_equipment == "hazmat_suit" else 0,
            "special_equipment_needed_none": 1 if special_equipment == "none" else 0,
        
            # === One-hot: Day of Week (7 categories, drop_first=False) ===
            "day_of_week_Friday": 1 if request_dayofweek == 4 else 0,
            "day_of_week_Monday": 1 if request_dayofweek == 0 else 0,
            "day_of_week_Saturday": 1 if request_dayofweek == 5 else 0,
            "day_of_week_Sunday": 1 if request_dayofweek == 6 else 0,
            "day_of_week_Thursday": 1 if request_dayofweek == 3 else 0,
            "day_of_week_Tuesday": 1 if request_dayofweek == 1 else 0,
            "day_of_week_Wednesday": 1 if request_dayofweek == 2 else 0,
        }

        # ========================================================
        # STEP 4: Create DataFrame and align with model features
        # ========================================================

        input_df = pd.DataFrame([input_dict])
    
        # Ensure all model features exist (fill missing with 0)
        for col in FEATURES:
            if col not in input_df.columns:
                input_df[col] = 0
    
        # Reorder columns to EXACTLY match training order
        input_df = input_df[FEATURES]

        # ========================================================
        # STEP 5: Make prediction
        # ========================================================

        prediction = float(model.predict(input_df)[0])
        prediction = round(prediction, 1)
        prediction = max(prediction, 3.0)  # Minimum realistic time

        # ========================================================
        # STEP 6: Display results
        # ========================================================

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric("⏱️ Predicted Retrieval Time", f"{prediction:.1f} min")

        with r2:
            if prediction < 60:
                level = "🟢 Fast"
            elif prediction < 100:
                level = "🟡 Moderate"
            elif prediction < 140:
                level = "🟠 Slow"
            else:
                level = "🔴 Very Slow"
            st.metric("Speed Category", level)

        with r3:
            hrs = int(prediction // 60)
            mins = int(prediction % 60)
            eta = f"{hrs}h {mins}m" if hrs > 0 else f"{mins}m"
            st.metric("🕐 ETA Duration", eta)

        # Result box
        box_cls = "result-box-high" if prediction >= 120 else "result-box"
        st.markdown(
            f"""
            <div class="{box_cls}">
            Estimated Retrieval Time:
            <br><br>
            <strong>{prediction:.1f} minutes</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Key drivers
        st.markdown("### 🔍 Key Prediction Drivers")
        st.markdown(f"""
        - Warehouse Zone: **{warehouse_zone}**
        - Aisles Selected: **{', '.join(map(str, aisle_numbers))}**
        - Distance From Dock: **{distance_from_dock} m**
        - Traffic Level: **{aisle_traffic}**
        - Active Retrievals: **{active_retrievals}**
        - Weight: **{total_weight} kg**
        - Number of Aisles: **{num_aisles}**
        - Peak Hour: **{'Yes' if is_peak_hour else 'No'}**
        - Operator Experience: **{operator_experience} years**
        """)

with tab2:

    with tab2:

        st.markdown("### 📊 EDA Dashboard")
        st.markdown(
            "All charts below match exactly the EDA done in **EDA.ipynb**. "
            "Same colours, same groupings, same logic."
        )

        # ── Pre-process df_raw to match EDA notebook state ──────
        # (Cell 21 — fill missing)
        df = df_raw.copy()
        for col in ["dock_queue_length","operator_shift_fatigue"]:
            df[col] = df[col].fillna(df[col].median())
        for col in ["aisle_traffic_level","fragile_handling"]:
            df[col] = df[col].fillna(df[col].mode()[0])

        # (Cell 23 — fix bool dtypes)
        for col in ["requires_forklift","fragile_handling"]:
            df[col] = df[col].map(
                {"True":1,"False":0,True:1,False:0}
            ).astype(float)

        TARGET    = "retrieval_time_minutes"
        dow_order = ["Monday","Tuesday","Wednesday",
                     "Thursday","Friday","Saturday","Sunday"]

        # ── KPI summary cards ────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Records",         f"{len(df):,}")
        k2.metric("Avg Retrieval Time",    f"{df[TARGET].mean():.1f} min")
        k3.metric("Median Retrieval Time", f"{df[TARGET].median():.1f} min")
        k4.metric("Max Retrieval Time",    f"{df[TARGET].max():.1f} min")
        k5.metric("Min Retrieval Time",    f"{df[TARGET].min():.1f} min")

        st.markdown("---")

        # ════════════════════════════════════════════════════════
        # CHART 1 — Target Variable Distribution
        # Source : EDA.ipynb Cell 10
        # ════════════════════════════════════════════════════════
        st.markdown("#### Chart 1 — Target Variable Distribution")

        t = df[TARGET]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Chart 1 — Target Variable: retrieval_time_minutes",
                     fontsize=13, fontweight="bold")

        # 1a. Raw histogram + KDE
        ax = axes[0]
        ax.hist(t, bins=60, color=BLUE, alpha=0.75, edgecolor="white", density=True)
        t.plot.kde(ax=ax, color=CORAL, linewidth=2)
        ax.axvline(t.mean(),   color=AMBER, linestyle="--", linewidth=1.5,
                   label=f"Mean   {t.mean():.0f} min")
        ax.axvline(t.median(), color=GREEN, linestyle="--", linewidth=1.5,
                   label=f"Median {t.median():.0f} min")
        ax.set_title("Raw Distribution")
        ax.set_xlabel("Retrieval time (min)"); ax.set_ylabel("Density")
        ax.legend(fontsize=9)

        # 1b. Log transformed
        ax    = axes[1]
        log_t = np.log1p(t)
        ax.hist(log_t, bins=60, color=TEAL, alpha=0.75, edgecolor="white", density=True)
        log_t.plot.kde(ax=ax, color=CORAL, linewidth=2)
        ax.set_title("Log-Transformed Distribution")
        ax.set_xlabel("log(1 + retrieval time)"); ax.set_ylabel("Density")
        ax.text(0.97, 0.97,
                f"Raw skew : {t.skew():.2f}\nLog skew : {log_t.skew():.2f}",
                transform=ax.transAxes, va="top", ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # 1c. Boxplot
        ax = axes[2]
        ax.boxplot(t, patch_artist=True, widths=0.5,
                   boxprops    =dict(facecolor=BLUE,  alpha=0.5),
                   medianprops =dict(color=CORAL,      linewidth=2),
                   whiskerprops=dict(color=GRAY),
                   capprops    =dict(color=GRAY),
                   flierprops  =dict(marker="o", color=AMBER,
                                     markersize=2, alpha=0.4))
        ax.set_title("Box Plot — Outlier Check")
        ax.set_ylabel("Retrieval time (min)"); ax.set_xticks([])
        pct99 = np.percentile(t, 99)
        ax.axhline(pct99, color=PURPLE, linestyle=":", linewidth=1.5,
                   label=f"P99 = {pct99:.0f} min")
        ax.legend(fontsize=9)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("---")

        # ════════════════════════════════════════════════════════
        # CHART 2 — Categorical Features vs Retrieval Time
        # Source : EDA.ipynb Cell 12
        # ════════════════════════════════════════════════════════
        st.markdown("#### Chart 2 — Categorical Features vs Retrieval Time (Median)")

        cat_features = ["warehouse_zone","shelf_level","storage_type",
                        "aisle_traffic_level","day_of_week",
                        "special_equipment_needed","requires_forklift"]
        colors_map   = {
            "warehouse_zone"          : BLUE,
            "shelf_level"             : TEAL,
            "storage_type"            : CORAL,
            "aisle_traffic_level"     : AMBER,
            "day_of_week"             : PURPLE,
            "special_equipment_needed": GREEN,
            "requires_forklift"       : GRAY,
        }

        # Restore string columns for groupby (needed after bool conversion)
        df_cat = df_raw.copy()
        for col in ["aisle_traffic_level","fragile_handling"]:
            df_cat[col] = df_cat[col].fillna(df_cat[col].mode()[0])

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle("Chart 2 — Categorical Features vs Retrieval Time (Median)",
                     fontsize=13, fontweight="bold")
        axes = axes.flatten()

        for idx, col in enumerate(cat_features):
            ax    = axes[idx]
            tmp   = df_cat[[col, TARGET]].dropna()
            order = ([d for d in dow_order if d in tmp[col].unique()]
                     if col == "day_of_week"
                     else tmp.groupby(col)[TARGET].median()
                             .sort_values(ascending=False).index.tolist())
            medians = tmp.groupby(col)[TARGET].median().reindex(order)
            counts  = tmp.groupby(col)[TARGET].count().reindex(order)
            bars    = ax.bar(range(len(order)), medians.values,
                             color=colors_map[col], alpha=0.75, edgecolor="white")
            ax.set_title(col.replace("_"," ").title())
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([str(o)[:10] for o in order],
                               rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("Median retrieval time (min)")
            for bar, cnt in zip(bars, counts.values):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.5,
                        f"n={cnt:,}", ha="center", va="bottom",
                        fontsize=7)

        axes[-1].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("---")

        # ════════════════════════════════════════════════════════
        # CHART 3 — Numerical Features vs Retrieval Time
        # Source : EDA.ipynb Cell 13
        # ════════════════════════════════════════════════════════
        st.markdown("#### Chart 3 — Numerical Features vs Retrieval Time (3,000 sample)")

        num_features = ["distance_from_dock_meters","active_retrievals_count",
                        "num_aisles_to_visit","dock_queue_length","total_weight_kg",
                        "operator_experience_years","num_packages","operator_shift_fatigue"]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle("Chart 3 — Numerical Features vs Retrieval Time (3,000 sample)",
                     fontsize=13, fontweight="bold")
        axes   = axes.flatten()
        sample = df.sample(3000, random_state=42)

        for idx, col in enumerate(num_features):
            ax   = axes[idx]
            s    = sample[[col, TARGET]].dropna()
            corr = df[[col, TARGET]].dropna().corr()[TARGET][col]
            c    = BLUE if corr >= 0 else CORAL
            ax.scatter(s[col], s[TARGET], alpha=0.15, s=8, color=c)
            z      = np.polyfit(s[col], s[TARGET], 1)
            x_line = np.linspace(s[col].min(), s[col].max(), 100)
            ax.plot(x_line, np.poly1d(z)(x_line), color=AMBER, linewidth=2)
            ax.set_xlabel(col.replace("_"," "), fontsize=8)
            ax.set_ylabel("Retrieval time (min)", fontsize=8)
            ax.set_title(f"{col.replace('_',' ')}  |  r = {corr:.3f}", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("---")

        # ════════════════════════════════════════════════════════
        # CHART 4 — Correlation Analysis
        # Source : EDA.ipynb Cell 15
        # ════════════════════════════════════════════════════════
        st.markdown("#### Chart 4 — Correlation Analysis")

        num_cols = ["distance_from_dock_meters","active_retrievals_count",
                    "dock_queue_length","num_aisles_to_visit","total_weight_kg",
                    "operator_experience_years","num_packages",
                    "operator_shift_fatigue","aisle_number","shift_hour", TARGET]

        corr_mat = df[num_cols].corr()
        mask     = np.triu(np.ones_like(corr_mat, dtype=bool))

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle("Chart 4 — Correlation Analysis", fontsize=13, fontweight="bold")

        sns.heatmap(corr_mat, mask=mask, ax=axes[0], cmap="RdBu_r", center=0,
                    annot=True, fmt=".2f", linewidths=0.4, linecolor="#EEEEEE",
                    annot_kws={"size":7}, vmin=-1, vmax=1, cbar_kws={"shrink":0.8})
        axes[0].set_title("Pearson Correlation Matrix")
        axes[0].tick_params(axis="x", rotation=45, labelsize=8)

        corr_target = corr_mat[TARGET].drop(TARGET).sort_values(key=abs, ascending=True)
        bar_colors  = [BLUE if v > 0 else CORAL for v in corr_target.values]
        axes[1].barh(range(len(corr_target)), corr_target.values,
                     color=bar_colors, alpha=0.8, edgecolor="white")
        axes[1].set_yticks(range(len(corr_target)))
        axes[1].set_yticklabels([c.replace("_"," ") for c in corr_target.index],
                                fontsize=9)
        axes[1].axvline(0, color=GRAY, linewidth=0.8)
        axes[1].set_xlabel("Pearson r with retrieval_time_minutes")
        axes[1].set_title("Feature Correlation with Target\n"
                          "(blue = positive, red = negative)")
        for i, v in enumerate(corr_target.values):
            axes[1].text(v+(0.005 if v>=0 else -0.005), i, f"{v:.3f}",
                         va="center", ha="left" if v>=0 else "right", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("---")

        # ════════════════════════════════════════════════════════
        # CHART 5 — Hidden Interaction Effects
        # Source : EDA.ipynb Cell 17
        # ════════════════════════════════════════════════════════
        st.markdown("#### Chart 5 — Hidden Interaction Effects")

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle("Chart 5 — Hidden Interaction Effects",
                     fontsize=13, fontweight="bold")

        # 5a. Zone × Shelf
        pivot1 = (df_raw.groupby(["warehouse_zone","shelf_level"])[TARGET]
                  .mean().unstack()
                  .reindex(["Zone_A","Zone_B","Zone_C","Zone_D"])[["Floor","Mid","High"]])
        sns.heatmap(pivot1, ax=axes[0], cmap="YlOrRd", annot=True, fmt=".0f",
                    linewidths=0.5, linecolor="white", annot_kws={"size":11},
                    cbar_kws={"label":"Mean retrieval time (min)"})
        axes[0].set_title("Zone × Shelf Level")

        # 5b. Traffic × Forklift
        pivot2 = (df_raw.groupby(["aisle_traffic_level","requires_forklift"])[TARGET]
                  .mean().unstack().reindex(["Low","Medium","High"]))
        sns.heatmap(pivot2, ax=axes[1], cmap="YlOrRd", annot=True, fmt=".0f",
                    linewidths=0.5, linecolor="white", annot_kws={"size":13},
                    cbar_kws={"label":"Mean retrieval time (min)"})
        axes[1].set_title("Traffic Level × Forklift Required")

        # 5c. Day × Storage type
        pivot3 = (df_raw.groupby(["day_of_week","storage_type"])[TARGET]
                  .mean().unstack().reindex(dow_order))
        sns.heatmap(pivot3, ax=axes[2], cmap="YlOrRd", annot=True, fmt=".0f",
                    linewidths=0.5, linecolor="white", annot_kws={"size":9},
                    cbar_kws={"label":"Mean retrieval time (min)"})
        axes[2].set_title("Day of Week × Storage Type")
        axes[2].tick_params(axis="x", rotation=30)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("---")

        # ════════════════════════════════════════════════════════
        # CHART 6 — Key Feature Distributions
        # Source : EDA.ipynb Cell 19
        # ════════════════════════════════════════════════════════
        st.markdown("#### Chart 6 — Key Feature Distributions")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Chart 6 — Key Feature Distributions",
                     fontsize=13, fontweight="bold")

        # 6a. Missing values
        ax   = axes[0,0]
        miss = (df_raw.isnull().mean()*100)
        miss = miss[miss > 0].sort_values(ascending=True)
        ax.barh(miss.index, miss.values, color=CORAL, alpha=0.8, edgecolor="white")
        ax.axvline(2, color=GRAY, linestyle="--", linewidth=1, label="2% threshold")
        ax.set_xlabel("Missing %"); ax.set_title("Missing Values by Column")
        ax.legend(fontsize=8)
        for i, v in enumerate(miss.values):
            ax.text(v+0.02, i, f"{v:.2f}%", va="center", fontsize=9)

        # 6b. Shift hour
        ax       = axes[0,1]
        hr_means = df[["shift_hour", TARGET]].groupby("shift_hour")[TARGET].mean()
        ax.bar(hr_means.index, hr_means.values, color=PURPLE,
               alpha=0.75, edgecolor="white")
        ax.axvspan(8,  11, alpha=0.12, color=AMBER, label="Morning peak (8–11)")
        ax.axvspan(16, 19, alpha=0.12, color=CORAL, label="Evening peak (16–19)")
        ax.set_xlabel("Hour of Day"); ax.set_ylabel("Mean retrieval time (min)")
        ax.set_title("Retrieval Time by Shift Hour"); ax.legend(fontsize=8)

        # 6c. Num aisles
        ax   = axes[0,2]
        nais = df.groupby("num_aisles_to_visit")[TARGET].mean()
        ax.bar(nais.index, nais.values, color=TEAL, alpha=0.8, edgecolor="white")
        ax.set_xlabel("Number of Aisles to Visit")
        ax.set_ylabel("Mean retrieval time (min)")
        ax.set_title("num_aisles_to_visit → Retrieval Time")

        # 6d. Weight distribution
        ax = axes[1,0]
        ax.hist(df["total_weight_kg"], bins=60, color=BLUE,
                alpha=0.75, edgecolor="white")
        ax.set_xlabel("Total Weight (kg)"); ax.set_ylabel("Count")
        ax.set_title("total_weight_kg (Log-Normal, Right-Skewed)")

        # 6e. Operator experience
        ax      = axes[1,1]
        exp_bin = pd.cut(df["operator_experience_years"],
                         bins=[0,2,5,10,15,25],
                         labels=["0–2yr","2–5yr","5–10yr","10–15yr","15–25yr"])
        exp_means = df.groupby(exp_bin, observed=True)[TARGET].mean()
        ax.bar(exp_means.index, exp_means.values, color=GREEN,
               alpha=0.8, edgecolor="white")
        ax.set_xlabel("Operator Experience")
        ax.set_ylabel("Mean retrieval time (min)")
        ax.set_title("Operator Experience → Retrieval Time")

        # 6f. Active retrievals by traffic level
        ax = axes[1,2]
        for lvl, color in zip(["Low","Medium","High"], [GREEN, AMBER, CORAL]):
            sub = df[df["aisle_traffic_level"]==lvl]["active_retrievals_count"].dropna()
            ax.hist(sub, bins=30, alpha=0.55, color=color,
                    label=lvl, edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Active Retrievals Count"); ax.set_ylabel("Count")
        ax.set_title("Active Retrievals by Traffic Level")
        ax.legend(title="Traffic Level", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()