import streamlit as st
import json
import hashlib
import os
import io
import numpy as np
import pandas as pd
from pandas import ExcelWriter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import newton

import firebase_admin
from firebase_admin import credentials, firestore

# ─── FIREBASE INITIALIZATION ───
@st.cache_resource
def init_firebase():
    key = "serviceAccountKey.json"
    if not os.path.exists(key):
        st.error(f"❌ {key} not found. Upload it to your working directory.")
        st.stop()
    cred = credentials.Certificate(key)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firebase()

# ─── WIRE IN YOUR CORE ENGINE ───
from finance_engine import run_financial_model_core

# ─── STREAMLIT PAGE CONFIG ───
st.set_page_config(page_title="Financial Model", layout="wide")
st.sidebar.image("logo.png", use_container_width=True)

page = st.sidebar.radio(
    "Go to",
    ["Market Setup", "Financial Model", "Consolidated View"],
    key="nav_main"
)

# ─── MODEL WRAPPER ───
@st.cache_data
def run_financial_model(sa: dict, sc: dict, adj_hash: str) -> pd.DataFrame:
    ui = st.session_state[adj_hash]
    adj = dict(sa)
    adj.update(ui)
    return run_financial_model_core(sa, sc, adj)

# ─── 0. SCHEMAS ───
ASSUMPTION_SCHEMA = {
    "base_fare_per_commuter_local":       float,
    "passenger_use_percentage":           float,
    "passengers_per_vehicle_per_day":     float,
    "transaction_fee_percentage":         float,
    "passenger_use_annual":               list,
    "vehicle_uptake_scaling":             list,
    "pos_device_cost_usd":                float,
    "pos_device_daily_rental_usd":        float,
    "working_days_per_month":             int,
    "weekend_days_per_month":             int,
    "weekend_use_relative_to_weekday":    float,
    "revenue_growth_after_period":        float,
    "nfc_tag_cost_per_unit_usd":          float,
    "tag_markup":                         float,
    "initial_tags_given_away":            int,
    "tag_purchase_batch_size":            int,
    "tag_purchase_buffer_percent":        float,
    "tag_purchase_lead_months":           int,
    "tag_replacement_percentage":         float,
    "office_rent_monthly_usd":            float,
    "connectivity_office_monthly_usd":    float,
    "connectivity_field_staff_monthly_usd": float,
    "staff_costs":                        list,
    "initial_marketing_cost_usd":         float,
    "ongoing_marketing_monthly_usd":      float,
    "third_party_license_fee_monthly_usd": float,
    "exchange_rate":                      float,
    "currency_depreciation_per_annum":    float,
    "inflation_rate":                     float,
    "debt_interest_rate":                 float,
    "cash_interest_rate":                 float,
    "bank_charges_percentage":            float,
    "wht_percentage":                     float,
    "corporate_tax_rate":                 float,
    "minority_interest":                  float,
    "ho_license_fee_percentage":          float,
    "depreciation_period_months":         int,
    "local_currency":                     str,
}

SCHEDULE_SCHEMA = {
    "vehicle_ramp_up_schedule":           list,
    "initial_marketing_payment_schedule": list,
}

# ─── VALIDATOR ───
def validate_dict(data: dict, schema: dict, name: str):
    missing = set(schema) - set(data.keys())
    extra   = set(data.keys()) - set(schema)
    if missing or extra:
        st.warning(f"🔍 {name} schema mismatch – missing: {missing}, extra: {extra}")
    else:
        st.success(f"✅ {name} schema OK")

# ─── HELPERS ───
def get_existing_countries():
    if not db:
        return []
    cols = [
        c.id for c in db.collections()
        if c.id.endswith("_data") and c.id != "app_data_simplified"
    ]
    names = sorted(c.replace("_data", "").replace("_", " ").title() for c in cols)
    return ["Select a country..."] + names

@st.cache_data
def get_country_data(country):
    if not db or country == "Select a country...":
        return None, None

    coll = country.lower().replace(" ", "_") + "_data"
    a_doc = db.collection(coll).document("assumptions").get()
    s_doc = db.collection(coll).document("schedules").get()

    if a_doc.exists and s_doc.exists:
        a = a_doc.to_dict()
        s = s_doc.to_dict()

        # ✅ Fallback: ensure marketing schedule exists
        if "initial_marketing_payment_schedule" not in s:
            s["initial_marketing_payment_schedule"] = [0.5, 0.3, 0.2]

        return a, s

    return None, None

def save_country_data(country, A, S):
    coll = country.lower().replace(" ", "_") + "_data"

    # ✅ Ensure the marketing schedule is present
    if "initial_marketing_payment_schedule" not in S:
        S["initial_marketing_payment_schedule"] = [0.5, 0.3, 0.2]

    db.collection(coll).document("assumptions").set(A)
    db.collection(coll).document("schedules").set(S)
    st.success(f"✅ Saved data for {country}")
    st.cache_data.clear()

def generate_vehicle_schedule(total, m1, m2, ramp_months):
    arr = np.zeros(60)
    arr[0], arr[1] = m1, m2
    inc = (total - m2) / ramp_months if ramp_months > 0 else 0
    curr = m2
    for i in range(2, 2 + ramp_months):
        if i < 60:
            curr += inc
            arr[i] = round(curr)
    if 2 + ramp_months < 60:
        arr[2 + ramp_months:] = total
    return arr.tolist()
# Block 9 Render (Part 1/2)
def render_model_dashboard(country: str, sa: dict, sc: dict):
    import io
    import numpy as np
    import pandas as pd
    from pandas import ExcelWriter
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.optimize import newton
    import json
    import hashlib

    validate_dict(sa, ASSUMPTION_SCHEMA, "Loaded assumptions")
    validate_dict(sc, SCHEDULE_SCHEMA, "Loaded schedules")

    ui = {}
    ui.update(st.session_state.get(f"adj_{country}", {}))
    ui["pilot_m1"] = sc["vehicle_ramp_up_schedule"][0]
    ui["pilot_m2"] = sc["vehicle_ramp_up_schedule"][1]

    h = hashlib.md5(json.dumps(ui, sort_keys=True).encode()).hexdigest()
    st.session_state[h] = ui
    if h not in st.session_state:
        st.warning("Default assumptions missing — restoring from saved country data.")
        st.session_state[h] = dict(sa)
    if h not in st.session_state or "base_fare_per_commuter_local" not in st.session_state[h]:
        st.warning("Restoring default assumptions from stored country data.")
        session_adj = dict(sa)
        session_adj["pilot_m1"] = sc["vehicle_ramp_up_schedule"][0]
        session_adj["pilot_m2"] = sc["vehicle_ramp_up_schedule"][1]
        st.session_state[h] = session_adj

    default_adj = st.session_state[h]
    df = run_financial_model(sa, sc, h)
    df["year"] = (df.index - 1) // 12 + 1
    df["ppe"] = df["capex_total"].cumsum() - df["depreciation"].cumsum()
    # Calculate true funding need before debt
    df["net_cash_flow"] = (
        df["net_income"]
        + df["depreciation"]
        - df["capex_total"]
    )

    df["cumulative_cashflow"] = df["net_cash_flow"].cumsum()
    peak_funding = -min(df["cumulative_cashflow"].min(), 0)
    agg = df.groupby("year").sum(numeric_only=True)
    view_df = df if st.session_state.get("freq_toggle", "Monthly") == "Monthly" else agg

    tab_static, tab_mc = st.tabs(["Static Model", "Monte Carlo Simulation"])

    with tab_static:
        st.subheader("Key Metrics (USD)")

        # Always compute metrics from monthly data to preserve accuracy
        rev_y5 = df[df["year"] == 5]["revenue_total"].sum()
        ni_y5 = df[df["year"] == 5]["net_income"].sum()


        # Corrected Peak Funding logic (true cash shortfall before debt)
        df["net_cash_flow"] = (
            df["net_income"]
            + df["depreciation"]
            - df["capex_total"]
        )

        df["cumulative_cashflow"] = df["net_cash_flow"].cumsum()
        peak_funding = -min(df["cumulative_cashflow"].min(), 0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Year 5 Revenue", f"${rev_y5:,.0f}")
        c2.metric("Year 5 Net Income", f"${ni_y5:,.0f}")
        c3.metric("Peak Funding", f"${peak_funding:,.0f}")

        st.subheader("Charts")
        st.line_chart(view_df[["revenue_total", "gross_profit", "ebitda", "net_income"]])
        st.line_chart(view_df[["cash", "debt_close"]])

        st.subheader("Financial Statements")
        st.markdown("### Income Statement")
        st.dataframe(
            view_df[["revenue_total", "cogs_tags", "gross_profit", "opex_total", "ebitda", "depreciation", "net_income"]]
            .style.format("${:,.0f}"),
            use_container_width=True
        )

        st.markdown("### Balance Sheet")
        st.dataframe(
            view_df[["cash", "ppe", "debt_close"]]
            .style.format("${:,.0f}"),
            use_container_width=True
        )

        st.markdown("### Cash Flow Statement")
        cashflow = pd.DataFrame(index=view_df.index)
        cashflow["Operating"] = view_df["net_income"] + view_df["depreciation"]
        cashflow["Investing"] = -view_df["capex_total"]
        cashflow["Financing"] = view_df["debt_draw"] - view_df["debt_rep"]
        cashflow["Net Cash Flow"] = cashflow.sum(axis=1)
        st.dataframe(cashflow.style.format("${:,.0f}"), use_container_width=True)

        st.subheader("Investor IRR")
        equity_injections = st.text_input("Equity Injections (e.g. 1:50000, 6:30000)", "1:50000")
        dividends         = st.text_input("Dividends (e.g. 36:30000, 48:50000)",       "60:100000")
        terminal_value    = st.number_input("Terminal Value (Month 60)", 0, 10000000, 500000)

        irr_cashflows = [0]*60
        for pair in equity_injections.split(","):
            if ":" in pair:
                m, amt = map(int, pair.strip().split(":"))
                irr_cashflows[m-1] -= amt
        for pair in dividends.split(","):
            if ":" in pair:
                m, amt = map(int, pair.strip().split(":"))
                irr_cashflows[m-1] += amt
        irr_cashflows[-1] += terminal_value

        def xirr(cf):
            def npv(rate):
                return sum(c / (1 + rate)**(i+1) for i, c in enumerate(cf))
            try:
                return newton(npv, 0.1)
            except:
                return float("nan")

        irr = xirr(irr_cashflows)
        st.metric("Investor IRR", f"{irr*100:.2f}%")

# Monte Carlo Sim Logic
# ───── Monte Carlo Simulation Tab ─────
    with tab_mc:
        st.subheader("Monte Carlo Simulation")

        country_doc = country.lower().replace(" ", "_") + "_data"
        saved_runs = db.collection(country_doc).list_documents()
        if saved_runs:
            run_ids = [doc.id for doc in saved_runs if doc.id.startswith("monte_carlo_runs_")]
            selected_run = st.selectbox("📁 Recall Past Run", options=["None"] + run_ids)
            if selected_run != "None":
                data = db.collection(country_doc).document(selected_run).get().to_dict()
                st.write("### Summary Statistics")
                st.json(data.get("summary", {}))
                st.write("### IRR Distribution")
                df = pd.DataFrame(data.get("results", []))
                st.dataframe(df.style.format({
                    "IRR": "{:.2f}%",
                    "Revenue": "${:,.0f}",
                    "EBITDA": "${:,.0f}",
                    "Net Income": "${:,.0f}",
                    "Peak Funding": "${:,.0f}"
                }), use_container_width=True)

        run_label = st.text_input("Label this run", "Run Label")
        num_simulations = st.slider("Number of Simulations", 100, 2000, 500, step=100)

        # Assumption Ranges UI
        st.markdown("### Assumption Ranges")
        mc_ranges = {}

        def mc_slider(label, key, minval, maxval, step=1.0):
            mc_ranges[key] = (
                st.number_input(f"{label} Min", value=minval, key=f"{key}_min"),
                st.number_input(f"{label} Max", value=maxval, key=f"{key}_max")
            )

        mc_slider("Base Fare", "base_fare_per_commuter_local", 80.0, 100.0)
        mc_slider("Transaction Fee %", "transaction_fee_percentage", 0.01, 0.05, step=0.01)
        mc_slider("Passengers/Veh/Day", "passengers_per_vehicle_per_day", 200, 400)
        mc_slider("POS Rent (USD/day)", "pos_device_daily_rental_usd", 3.0, 7.0)
        mc_slider("Marketing (USD/mo)", "ongoing_marketing_monthly_usd", 1000.0, 10000.0)
        mc_slider("Tag Markup", "tag_markup", 0.05, 0.3)
        mc_slider("Tag Cost USD", "nfc_tag_cost_per_unit_usd", 0.5, 2.0)
        mc_slider("Tag Replacement %", "tag_replacement_percentage", 0.01, 0.05)
        mc_slider("Working Days/Mo", "working_days_per_month", 20, 23)
        mc_slider("Weekend Days/Mo", "weekend_days_per_month", 4, 10)
        mc_slider("Weekend Use % Rel", "weekend_use_relative_to_weekday", 0.2, 0.5)
        for i in range(5):
            mc_ranges[f"passenger_use_annual_{i}"] = (
                st.number_input(f"Year {i+1} Use % Min", value=0.4, key=f"use_min_{i}"),
                st.number_input(f"Year {i+1} Use % Max", value=0.7, key=f"use_max_{i}")
            )
            mc_ranges[f"vehicle_uptake_scaling_{i}"] = (
                st.number_input(f"Year {i+1} Uptake % Min", value=0.8, key=f"uptake_min_{i}"),
                st.number_input(f"Year {i+1} Uptake % Max", value=1.2, key=f"uptake_max_{i}")
            )
        mc_slider("Bank Charges %", "bank_charges_percentage", 0.002, 0.02)
        mc_slider("Cash Interest %", "cash_interest_rate", 0.01, 0.05)
        mc_slider("Debt Interest %", "debt_interest_rate", 0.08, 0.18)
        mc_slider("Inflation Rate", "inflation_rate", 0.02, 0.10)
        mc_slider("FX Depreciation %", "currency_depreciation_per_annum", 0.02, 0.10)

        mc_ranges["staff_costs"] = {}
        for s in default_adj["staff_costs"]:
            role = s["role"]
            mc_ranges["staff_costs"][role] = {
                "count": (
                    st.number_input(f"{role} Count Min", value=s["count"], key=f"{role}_count_min"),
                    st.number_input(f"{role} Count Max", value=s["count"] + 2, key=f"{role}_count_max")
                ),
                "salary": (
                    st.number_input(f"{role} Salary Min", value=s["salary_usd"] * 0.8, key=f"{role}_salary_min"),
                    st.number_input(f"{role} Salary Max", value=s["salary_usd"] * 1.2, key=f"{role}_salary_max")
                )
            }

        equity_range = st.slider("Equity Injection (Month 1)", 10000, 500000, (20000, 80000), step=5000)
        dividend_range = st.slider("Dividend (Month 60)", 0, 300000, (0, 200000), step=10000)
        terminal_range = st.slider("Terminal Value (Month 60)", 0, 1000000, (300000, 700000), step=10000)

        if st.button("Run Monte Carlo Simulation"):
            results = []
            progress = st.progress(0)

            for i in range(num_simulations):
                sampled = dict(default_adj)
                for key, val in mc_ranges.items():
                    if key == "staff_costs": continue
                    if key.startswith("passenger_use_annual_"):
                        idx = int(key.split("_")[-1])
                        sampled.setdefault("passenger_use_annual", [0]*5)[idx] = np.random.uniform(*val)
                    elif key.startswith("vehicle_uptake_scaling_"):
                        idx = int(key.split("_")[-1])
                        sampled.setdefault("vehicle_uptake_scaling", [1]*5)[idx] = np.random.uniform(*val)
                    else:
                        sampled[key] = np.random.uniform(*val)

                sampled["staff_costs"] = []
                for role, bounds in mc_ranges["staff_costs"].items():
                    cmin, cmax = bounds["count"]
                    smin, smax = bounds["salary"]
                    sampled["staff_costs"].append({
                        "role": role,
                        "count": int(np.random.randint(cmin, cmax+1)),
                        "salary_usd": float(np.random.uniform(smin, smax))
                    })

                eq = int(np.random.uniform(*equity_range))
                div = int(np.random.uniform(*dividend_range))
                tv = int(np.random.uniform(*terminal_range))

                df_sim = run_financial_model_core(sa, sc, sampled)
                df_sim["cash_pre_debt"] = df_sim["cash"] - (df_sim["debt_draw"] - df_sim["debt_rep"])

                irr_cf = [0] * 60
                irr_cf[0] = -eq
                irr_cf[-1] = div + tv

                def xirr(cf):
                    def npv(rate): return sum(c / (1 + rate) ** (i + 1) for i, c in enumerate(cf))
                    try: return newton(npv, 0.1)
                    except: return float("nan")

                results.append({
                    "IRR": xirr(irr_cf) * 100,
                    "Revenue": df_sim["revenue_total"].sum(),
                    "EBITDA": df_sim["ebitda"].sum(),
                    "Net Income": df_sim["net_income"].sum(),
                    "Peak Funding": -df_sim["cash_pre_debt"].min(),
                    "Equity": eq,
                    "Dividend": div,
                    "Terminal": tv
                })
                progress.progress((i + 1) / num_simulations)

            df_res = pd.DataFrame(results)
            st.success("✅ Simulations complete!")

            st.subheader("📊 Investor IRR Distribution")
            fig, ax = plt.subplots()
            sns.boxplot(data=df_res, y="IRR", ax=ax)
            ax.set_ylabel("IRR (%)")
            st.pyplot(fig)

            for metric in ["Revenue", "EBITDA", "Net Income", "Peak Funding"]:
                st.subheader(f"📊 {metric} Distribution")
                fig, ax = plt.subplots()
                sns.boxplot(data=df_res, y=metric, ax=ax)
                ax.set_ylabel(f"{metric} (USD)")
                st.pyplot(fig)

            summary_df = df_res.describe(percentiles=[0.025, 0.5, 0.975]).T
            formatters = {
                "IRR": "{:,.2f}%", "Revenue": "${:,.0f}", "EBITDA": "${:,.0f}",
                "Net Income": "${:,.0f}", "Peak Funding": "${:,.0f}",
                "Equity": "${:,.0f}", "Dividend": "${:,.0f}", "Terminal": "${:,.0f}"
            }
            st.subheader("📋 Summary Statistics")
            st.dataframe(summary_df.style.format(formatters), use_container_width=True)

            buffer = io.BytesIO()
            with ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df_res.to_excel(writer, sheet_name="Simulation Results", index=False)
                flat_ranges = {}
                for k, v in mc_ranges.items():
                    if isinstance(v, tuple):
                        flat_ranges[k] = {"Min": v[0], "Max": v[1]}
                    elif isinstance(v, dict):
                        for role, bounds in v.items():
                            flat_ranges[f"{role} Count"] = {"Min": bounds["count"][0], "Max": bounds["count"][1]}
                            flat_ranges[f"{role} Salary"] = {"Min": bounds["salary"][0], "Max": bounds["salary"][1]}
                pd.DataFrame(flat_ranges).T.to_excel(writer, sheet_name="Assumption Ranges")
            st.download_button("📥 Download Results", buffer.getvalue(), f"{country}_monte_carlo.xlsx")

            if run_label:
                run_doc = db.collection(country_doc).document(f"monte_carlo_runs_{run_label}")
                run_doc.set({
                    "label": run_label,
                    "created_at": firestore.SERVER_TIMESTAMP,
                    "results": df_res.to_dict("records"),
                    "summary": summary_df.to_dict(),
                    "inputs": {k: (v if isinstance(v, tuple) else str(v)) for k, v in mc_ranges.items()}
                })
                st.success(f"✅ Run '{run_label}' saved to Firestore.")
# ─── 10a. MARKET SETUP PAGE ───────────────────────────────────────────────────
def render_setup_page():
    st.header("Market Setup & Assumptions")
    countries = get_existing_countries()
    action = st.radio(
        "Action",
        ["Create New Market", "View/Edit Existing Market"],
        key="mk_action",
    )

    # load existing when editing
    assumptions, schedules = {}, {}
    country = None

    if action == "View/Edit Existing Market":
        if len(countries) <= 1:
            st.warning("No markets found – create one first.")
            return

        country = st.selectbox("Select Country", countries, key="mk_country")

        if country and country != "Select a country...":
            # ✅ Optional Delete Panel
            with st.expander("⚠️ Delete Country Data", expanded=False):
                st.markdown(f"Type `DELETE` below to delete all data for **{country}**.")
                confirm = st.text_input("Confirm Deletion", key="delete_confirm")

                if confirm == "DELETE":
                    coll = country.lower().replace(" ", "_") + "_data"
                    docs = db.collection(coll).list_documents()
                    for doc in docs:
                        doc.delete()
                    db.collection(coll).document("assumptions").delete()
                    db.collection(coll).document("schedules").delete()

                    st.success(f"✅ {country} deleted successfully.")
                    st.cache_data.clear()
                    st.stop()

            # ✅ Load data
            assumptions, schedules = get_country_data(country)

            # ✅ Apply fallback before validation
            if schedules is not None and "initial_marketing_payment_schedule" not in schedules:
                schedules["initial_marketing_payment_schedule"] = [0.5, 0.3, 0.2]
                            # Save patched version back to Firestore
                coll = country.lower().replace(" ", "_") + "_data"
                db.collection(coll).document("schedules").set(schedules)
                st.info("📌 Missing 'initial_marketing_payment_schedule' added and saved.")

            # ✅ Validate assumptions and schedules
            if assumptions:
                validate_dict(assumptions, ASSUMPTION_SCHEMA, "Loaded assumptions")
            if schedules:
                validate_dict(schedules, SCHEDULE_SCHEMA, "Loaded schedules")

            if not assumptions:
                return

    else:
        country = st.text_input("New Country Name", key="mk_new_country")

    # currency code
    if action == "Create New Market":
        curr = st.text_input("Local Currency Code", "USD", key="mk_curr")
    else:
        curr = assumptions.get("local_currency", "USD")
        st.markdown(f"**Currency:** {curr}")

    defaults = assumptions or {}
    form = st.form("mk_form")

    # A collects all your assumption inputs
    A = {}

    with form:
        # ── Vehicle Ramp-up ──
        st.subheader("Vehicle Ramp-up")
        total = form.number_input(
            "Total Vehicle Population",
            value=int(defaults.get("total_vehicle_population", 20000)),
            step=1000,
            key="mk_total",
        )
        p1 = form.number_input("Pilot Month 1", value=int(defaults.get("pilot_m1", 300)), step=10, key="mk_p1")
        p2 = form.number_input("Pilot Month 2", value=int(defaults.get("pilot_m2", 300)), step=10, key="mk_p2")
        ramp = form.number_input(
            "Ramp-up Months",
            value=int(defaults.get("ramp_up_months", 22)),
            min_value=1, max_value=60, step=1,
            key="mk_ramp",
        )

        # ── Revenue & Growth ──
        st.subheader("Revenue & Growth")
        A["base_fare_per_commuter_local"] = form.number_input(
            "Base Fare per Commuter (Local)",
            value=float(defaults.get("base_fare_per_commuter_local", 83.0)),
            step=0.1,
            key="mk_base_fare",
        )
        A["passenger_use_percentage"] = (
            form.number_input(
                "Base Passenger Use %",
                value=float(defaults.get("passenger_use_percentage", 0.55)) * 100,
                min_value=0.0, max_value=100.0, step=0.1,
                key="mk_use",
            ) / 100
        )
        A["passengers_per_vehicle_per_day"] = form.number_input(
            "Passengers/Veh/Day",
            value=float(defaults.get("passengers_per_vehicle_per_day", 300.0)),
            step=1.0,
            key="mk_ppd",
        )
        A["transaction_fee_percentage"] = (
            form.number_input(
                "Transaction Fee %",
                value=float(defaults.get("transaction_fee_percentage", 0.025)) * 100,
                min_value=0.0, max_value=100.0, step=0.1,
                key="mk_fee",
            ) / 100
        )

        # ── Annual Use & Uptake (5-year) ──
        st.subheader("Annual Use & Uptake (5-year)")
        default_use = defaults.get("passenger_use_annual", [0.55] * 5)
        default_upt = defaults.get("vehicle_uptake_scaling", [1.0] * 5)
        A["passenger_use_annual"], A["vehicle_uptake_scaling"] = [], []
        for i in range(5):
            A["passenger_use_annual"].append(
                form.number_input(
                    f"Year {i+1} Use %",
                    value=default_use[i] * 100,
                    min_value=0.0, max_value=100.0, step=0.1,
                    key=f"mk_pua_{i}",
                ) / 100
            )
            A["vehicle_uptake_scaling"].append(
                form.number_input(
                    f"Year {i+1} Uptake %",
                    value=default_upt[i] * 100,
                    min_value=0.0, max_value=200.0, step=1.0,
                    key=f"mk_vus_{i}",
                ) / 100
            )

        # ── POS Device Costs ──
        st.subheader("POS Device")
        A["pos_device_cost_usd"] = form.number_input(
            "POS Device Cost (USD)",
            value=float(defaults.get("pos_device_cost_usd", 500.0)),
            step=10.0, key="mk_pos_cost",
        )
        A["pos_device_daily_rental_usd"] = form.number_input(
            "POS Device Daily Rental (USD)",
            value=float(defaults.get("pos_device_daily_rental_usd", 5.0)),
            step=0.5, key="mk_pos_rental",
        )

        # ── Operational Days ──
        st.subheader("Operational Days")
        A["working_days_per_month"] = form.number_input(
            "Working Days/Mo",
            value=int(defaults.get("working_days_per_month", 22)),
            min_value=0, max_value=31, step=1, key="mk_work_days",
        )
        A["weekend_days_per_month"] = form.number_input(
            "Weekend Days/Mo",
            value=int(defaults.get("weekend_days_per_month", 8)),
            min_value=0, max_value=31, step=1, key="mk_week_days",
        )
        A["weekend_use_relative_to_weekday"] = (
            form.number_input(
                "Weekend Use % relative",
                value=float(defaults.get("weekend_use_relative_to_weekday", 0.3)) * 100,
                min_value=0.0, max_value=100.0, step=0.1,
                key="mk_week_use",
            ) / 100
        )
        A["revenue_growth_after_period"] = (
            form.number_input(
                "Rev Growth % after 5y",
                value=float(defaults.get("revenue_growth_after_period", 0.0)) * 100,
                min_value=0.0, max_value=100.0, step=0.1,
                key="mk_rev_growth",
            ) / 100
        )

        # ── Tag Sales ──
        st.subheader("Tag Sales")
        A["nfc_tag_cost_per_unit_usd"] = form.number_input(
            "NFC Tag Cost /unit (USD)",
            value=float(defaults.get("nfc_tag_cost_per_unit_usd", 1.0)),
            step=0.1, key="mk_tag_cost",
        )
        A["tag_markup"] = (
            form.number_input(
                "Tag Markup %",
                value=float(defaults.get("tag_markup", 0.2)) * 100,
                min_value=0.0, max_value=100.0, step=0.1,
                key="mk_tag_markup",
            ) / 100
        )
        A["initial_tags_given_away"] = form.number_input(
            "Initial Free Tags",
            value=int(defaults.get("initial_tags_given_away", 500000)),
            step=1, key="mk_free_tags",
        )
        A["tag_purchase_batch_size"] = form.number_input(
            "Tag Purchase Batch Size",
            value=int(defaults.get("tag_purchase_batch_size", 10000)),
            step=100, key="mk_tag_batch",
        )
        A["tag_purchase_buffer_percent"] = (
            form.number_input(
                "Tag Purchase Buffer %",
                value=float(defaults.get("tag_purchase_buffer_percent", 0.1)) * 100,
                min_value=0.0, max_value=100.0, step=0.1,
                key="mk_tag_buffer",
            ) / 100
        )
        A["tag_purchase_lead_months"] = form.number_input(
            "Tag Purchase Lead (Mo)",
            value=int(defaults.get("tag_purchase_lead_months", 2)),
            min_value=0, max_value=12, step=1, key="mk_tag_lead",
        )
        A["tag_replacement_percentage"] = (
            form.number_input(
                "Tag Replacement % (Annual)",
                value=float(defaults.get("tag_replacement_percentage", 0.05)) * 100,
                min_value=0.0, max_value=100.0, step=0.1,
                key="mk_tag_repl",
            ) / 100
        )

        # ── OpEx & Staffing ──
        st.subheader("OpEx & Staffing")
        A["office_rent_monthly_usd"] = form.number_input(
            "Office Rent (USD/Mo)",
            value=float(defaults.get("office_rent_monthly_usd", 10000.0)),
            step=100.0, key="mk_office_rent",
        )
        A["connectivity_office_monthly_usd"] = form.number_input(
            "Office Connectivity (USD/Mo)",
            value=float(defaults.get("connectivity_office_monthly_usd", 5000.0)),
            step=50.0, key="mk_conn_off",
        )
        A["connectivity_field_staff_monthly_usd"] = form.number_input(
            "Field-Staff Connectivity (USD/Mo)",
            value=float(defaults.get("connectivity_field_staff_monthly_usd", 50.0)),
            step=5.0, key="mk_conn_field",
        )

        # Staff roles — defaults only if creating a new market
        default_roles = [
            {"role": "Support Desk", "count": 2, "salary_usd": 800},
            {"role": "Field Agent", "count": 5, "salary_usd": 500},
            {"role": "Sacco Liaison", "count": 3, "salary_usd": 600}
        ]
        A["staff_costs"] = defaults.get("staff_costs", default_roles)

        st.markdown("#### Staffing Breakdown")
        for i, s in enumerate(A["staff_costs"]):
            cnt = form.number_input(
                f"{s['role']} Count",
                value=int(s.get("count", 0)), min_value=0, step=1,
                key=f"mk_staff_cnt_{i}_{s['role']}"
            )
            sal = form.number_input(
                f"{s['role']} Salary (USD)",
                value=float(s.get("salary_usd", 0.0)), min_value=0.0, step=100.0,
                key=f"mk_staff_sal_{i}_{s['role']}"
            )
            A["staff_costs"][i] = {"role": s["role"], "count": cnt, "salary_usd": sal}

        # ── Marketing Assumptions ──
        st.subheader("Marketing Assumptions")
        A["initial_marketing_cost_usd"] = form.number_input(
            "Initial Marketing (USD)",
            value=float(defaults.get("initial_marketing_cost_usd", 50000.0)),
            step=1000.0, key="mk_init_marketing",
        )
        A["ongoing_marketing_monthly_usd"] = form.number_input(
            "Ongoing Marketing (USD/Mo)",
            value=float(defaults.get("ongoing_marketing_monthly_usd", 5000.0)),
            step=100.0, key="mk_ongoing_marketing",
        )

        # ── Financial & Tax ──
        st.subheader("Financial & Tax")
        A["exchange_rate"] = form.number_input(
            f"Exchange Rate ({curr}→USD)",
            value=float(defaults.get("exchange_rate", 1.0)), step=0.001, key="mk_fx"
        )
        A["currency_depreciation_per_annum"] = (
            form.number_input(
                "Annual FX Depreciation %",
                value=float(defaults.get("currency_depreciation_per_annum", 0.0)) * 100,
                min_value=0.0, max_value=100.0, step=0.1,
                key="mk_fx_dep",
            ) / 100
        )
        A["inflation_rate"] = (
            form.number_input(
                "Annual Inflation %",
                value=float(defaults.get("inflation_rate", 0.0)) * 100,
                min_value=0.0, max_value=100.0, step=0.1,
                key="mk_infl",
            ) / 100
        )
        A["debt_interest_rate"] = (
            form.number_input(
                "Debt Interest %",
                value=float(defaults.get("debt_interest_rate", 0.0)) * 100,
                min_value=0.0, max_value=100.0, step=0.1,
                key="mk_debt_int",
            ) / 100
        )
        A["cash_interest_rate"] = (
            form.number_input(
                "Cash Interest %",
                value=float(defaults.get("cash_interest_rate", 0.0)) * 100,
                min_value=0.0, max_value=100.0, step=0.1,
                key="mk_cash_int",
            ) / 100
        )
        A["bank_charges_percentage"] = (
            form.number_input(
                "Bank Charges %",
                value=float(defaults.get("bank_charges_percentage", 0.0)) * 100,
                min_value=0.0, max_value=100.0, step=0.1,
                key="mk_bank_ch",
            ) / 100
        )
        A["wht_percentage"] = (
            form.number_input(
                "Withholding Tax %",
                value=float(defaults.get("wht_percentage", 0.0)) * 100,
                min_value=0.0, max_value=100.0, step=0.1,
                key="mk_wht",
            ) / 100
        )
        A["corporate_tax_rate"] = (
            form.number_input(
                "Corporate Tax Rate %",
                value=float(defaults.get("corporate_tax_rate", 0.0)) * 100,
                min_value=0.0, max_value=100.0, step=0.1,
                key="mk_corp_tax",
            ) / 100
        )
        A["minority_interest"] = (
            form.number_input(
                "Minority Interest %",
                value=float(defaults.get("minority_interest", 0.0)) * 100,
                min_value=0.0, max_value=100.0, step=0.1,
                key="mk_min_int",
            ) / 100
        )
        A["ho_license_fee_percentage"] = (
            form.number_input(
                "HO License Fee %",
                value=float(defaults.get("ho_license_fee_percentage", 0.0)) * 100,
                min_value=0.0, max_value=100.0, step=0.1,
                key="mk_ho_fee",
            ) / 100
        )
        A["depreciation_period_months"] = form.number_input(
            "Depreciation Period (Mo)",
            value=int(defaults.get("depreciation_period_months", 36)),
            min_value=1, max_value=120, step=1,
            key="mk_depr",
        )
        A["third_party_license_fee_monthly_usd"] = form.number_input(
            "3rd-Party License Fee (USD/Mo)",
            value=float(defaults.get("third_party_license_fee_monthly_usd", 0.0)),
            step=50.0, key="mk_third_party_license",
        )

        # the submit button
        submitted = form.form_submit_button("Save Assumptions")

    # end with form

    if submitted:
        if not country or not country.strip():
            st.error("Please enter a country name before saving.")
        else:
            S = {
                "vehicle_ramp_up_schedule": generate_vehicle_schedule(total, p1, p2, ramp)
            }
            if "initial_marketing_payment_schedule" in schedules:
                S["initial_marketing_payment_schedule"] = schedules["initial_marketing_payment_schedule"]
            A["local_currency"] = curr
            validate_dict(A, ASSUMPTION_SCHEMA, "Assumptions")
            validate_dict(S, SCHEDULE_SCHEMA, "Schedules")
        save_country_data(country, A, S)
# ─── 10b. ROUTER & CONSOLIDATED VIEW ─────────────────────────────────────────
# (make sure you have only one set_page_config at the top of the file)


if page == "Market Setup":
    render_setup_page()

elif page == "Financial Model":
    countries = get_existing_countries()
    if len(countries) <= 1:
        st.warning("No markets – create one first.")
    else:
        sel = st.selectbox("Select a Market", countries, key="mod_country")
        if sel != "Select a country...":
            sa, sc = get_country_data(sel)
            if sa and sc:
                # ✅ fallback now handled inside get_country_data()
                validate_dict(sa, ASSUMPTION_SCHEMA, "Assumptions")
                validate_dict(sc, SCHEDULE_SCHEMA,   "Schedules")
                render_model_dashboard(sel, sa, sc)

elif page == "Consolidated View":
    st.header("📋 Consolidated Financial View")

    # 1) Select Markets
    markets = get_existing_countries()[1:]
    selected = st.multiselect(
        "Select markets to include",
        options=markets,
        default=markets
    )

    if not selected:
        st.warning("No markets selected.")
    else:
        consolidated = {}
        for country in selected:
            sa, sc = get_country_data(country)
            if not sa or not sc:
                continue
            adj = dict(sa)
            adj["pilot_m1"] = sc["vehicle_ramp_up_schedule"][0]
            adj["pilot_m2"] = sc["vehicle_ramp_up_schedule"][1]
            df = run_financial_model_core(sa, sc, adj)

            df["net_cash_flow"] = df["net_income"] + df["depreciation"] - df["capex_total"]
            df["cumulative_cashflow"] = df["net_cash_flow"].cumsum()
            df["peak_funding_true"] = -df["cumulative_cashflow"].cummin()

            consolidated[country] = df

        # 2) Summary Table
        rows = []
        for c, df in consolidated.items():
            rows.append({
                "Country": c,
                "5-Yr Revenue": df["revenue_total"].sum(),
                "5-Yr Net Income": df["net_income"].sum(),
                "Peak Funding": df["peak_funding_true"].max()
            })
        summary_df = pd.DataFrame(rows).set_index("Country")

        st.subheader("Key Metrics by Market")
        st.dataframe(
            summary_df.style.format({
                "5-Yr Revenue": "${:,.0f}",
                "5-Yr Net Income": "${:,.0f}",
                "Peak Funding": "${:,.0f}"
            }),
            use_container_width=True
        )

        # 3) Combined Charts
        st.subheader("Revenue Over Time")
        st.line_chart({c: df["revenue_total"] for c, df in consolidated.items()})

        st.subheader("Net Income Over Time")
        st.line_chart({c: df["net_income"] for c, df in consolidated.items()})

        st.subheader("Cash Balance Over Time")
        st.line_chart({c: df["cash"] for c, df in consolidated.items()})
