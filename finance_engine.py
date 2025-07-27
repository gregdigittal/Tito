import pandas as pd
import numpy as np

def run_financial_model_core(sa: dict, sc: dict, adj: dict) -> pd.DataFrame:
    """
    Core financial model: takes setup assumptions (sa),
    ramp schedules (sc) and per-run adjustments (adj),
    and returns a 60-month DataFrame with full P&L, cash/debt, CapEx, etc.
    """
    months = 60
    df = pd.DataFrame(index=range(1, months + 1))

    # --- FX series (annual depreciation) ---
    rate = adj["exchange_rate"]
    fx = [rate * (1 + adj.get("currency_depreciation_per_annum", 0.0))**(m // 12)
          for m in range(months)]
    df["fx_rate"] = fx

    # --- per-month interest rates ---
    dr = adj["debt_interest_rate"] / 12
    cr = adj["cash_interest_rate"] / 12

    # --- flatten 5-year vectors to per-month ---
    vs = np.repeat(adj["vehicle_uptake_scaling"], 12)
    ps = np.repeat(adj["passenger_use_annual"],   12)

    # --- active vehicles ramp (first two months overridden by pilot) ---
    base = np.array(sc["vehicle_ramp_up_schedule"][:months], dtype=float)
    base[0], base[1] = adj["pilot_m1"], adj["pilot_m2"]
    df["active_vehicles"] = base * vs

    # --- fare (USD) escalated by inflation annually ---
    fare = adj["base_fare_per_commuter_local"] / adj["exchange_rate"]
    fares = []
    for m in range(months):
        if m > 0 and m % 12 == 0:
            fare *= (1 + adj["inflation_rate"])
        fares.append(fare)
    df["fare_usd"] = fares

    # --- revenues ---
    # --- revenue inputs ---
    daily = df["active_vehicles"] * adj["passengers_per_vehicle_per_day"] * ps
    days  = (
        adj["working_days_per_month"]
        + adj["weekend_days_per_month"] * adj["weekend_use_relative_to_weekday"]
    )

    # ✅ Add boardings for dashboard table
    df["delta_boardings"] = daily.diff().fillna(daily.iloc[0])
    df["total_boardings"] = daily * days
    df["rev_txn"]    = daily * days * df["fare_usd"] * adj["transaction_fee_percentage"]
    df["rev_rental"] = df["active_vehicles"] * adj["pos_device_daily_rental_usd"] * adj["working_days_per_month"]

    # --- tag sales revenue ---
    delta = daily.diff().fillna(daily.iloc[0])
    tags_issued_trailing = delta * 1.05  # trailing = based on current delta
    cum = tags_issued_trailing.cumsum()

    sold = (
        np.maximum(0, cum - adj["initial_tags_given_away"])
      - np.maximum(0, cum.shift(1).fillna(0) - adj["initial_tags_given_away"])
    )
    price_tag = adj["nfc_tag_cost_per_unit_usd"] * (1 + adj["tag_markup"])
    df["rev_tags"] = sold * price_tag


    # --- tag COGS (lead-based logic) ---
    delta_forward = delta.shift(-adj["tag_purchase_lead_months"], fill_value=0)
    tags_to_order = delta_forward * 1.05
    df["cogs_tags"] = tags_to_order * adj["nfc_tag_cost_per_unit_usd"]

    # --- revenue total ---
    df["revenue_total"] = df[["rev_txn", "rev_rental", "rev_tags"]].sum(axis=1)

    df["gross_profit"]  = df["revenue_total"] - df["cogs_tags"]

    # --- OPEX ---
    df["opex_salaries"] = sum(s["count"] * s["salary_usd"] for s in adj["staff_costs"])

    # Marketing: spread initial spend across schedule, then ongoing
    mkt = np.full(months, adj["ongoing_marketing_monthly_usd"], dtype=float)
    sched = sc.get("initial_marketing_payment_schedule", [])
    if sched:
        frac = np.array(sched, dtype=float)
        mkt[: frac.size] = adj["initial_marketing_cost_usd"] * frac
    df["opex_marketing"] = mkt

    df["opex_bank"]        = df["revenue_total"] * adj["bank_charges_percentage"]
    df["opex_office_rent"] = adj["office_rent_monthly_usd"]
    df["opex_license"]     = adj.get("third_party_license_fee_monthly_usd", 0.0)
    df["opex_conn_office"] = adj["connectivity_office_monthly_usd"]
    df["opex_conn_field"]  = (
        sum(s["count"] for s in adj["staff_costs"] if "Sacco" in s["role"])
        * adj["connectivity_field_staff_monthly_usd"]
    )
    df["opex_pos_rental"]      = df["active_vehicles"] * adj["pos_device_daily_rental_usd"] * adj["working_days_per_month"]
    df["opex_nfc_replacement"] = (
        cum.shift(1, fill_value=0)
        * (adj["tag_replacement_percentage"] / 12)
        * adj["nfc_tag_cost_per_unit_usd"]
    )

    df["opex_total"] = df[
        [
            "opex_salaries","opex_marketing","opex_bank","opex_office_rent",
            "opex_license","opex_conn_office","opex_conn_field",
            "opex_pos_rental","opex_nfc_replacement"
        ]
    ].sum(axis=1)
    df["ebitda"] = df["gross_profit"] - df["opex_total"]

    # --- CapEx & Depreciation ---
    init_cap = sum(item["cost_usd"] * item["count"] for item in sa.get("initial_capex_usd", []))
    incr     = df["active_vehicles"].diff().fillna(df["active_vehicles"].iloc[0])
    df["capex_pos"]        = incr * adj.get("pos_cost_usd", 0.0)
    df["capex_pos_device"] = incr * adj["pos_device_cost_usd"]
    df["capex_initial"]    = 0.0
    df.at[1, "capex_initial"] = init_cap
    df["capex_total"] = df[["capex_initial","capex_pos","capex_pos_device"]].sum(axis=1)
    df["depreciation"] = (
        df["capex_total"]
          .rolling(adj["depreciation_period_months"], min_periods=1)
          .sum()
        / adj["depreciation_period_months"]
    )

    # --- Cash / Debt Loop ---
    df[[
      "cash","debt_open","debt_draw","debt_rep",
      "debt_close","int_inc","int_exp","ebt","tax","net_income"
    ]] = 0.0
    loss_cf = 0.0
    for t in df.index:
        oc = df.at[t-1, "cash"]       if t > 1 else 0.0
        od = df.at[t-1, "debt_close"] if t > 1 else 0.0

        # interest
        df.at[t, "int_exp"] = od * dr
        df.at[t, "int_inc"] = oc * cr if oc > 0 else 0.0

        ebit = df.at[t, "ebitda"] - df.at[t, "depreciation"]
        lic  = (df.at[t,"rev_txn"] + df.at[t,"rev_rental"]) * adj["ho_license_fee_percentage"]
        ebt  = ebit - df.at[t,"int_exp"] + df.at[t,"int_inc"] - lic

        wht = ebt * adj["wht_percentage"] if ebt > 0 else 0.0
        ti  = ebt - wht
        if ti < 0:
            loss_cf -= ti
            taxable = 0.0
        else:
            taxable = max(0.0, ti - loss_cf)
            loss_cf = max(0.0, loss_cf - ti)

        tax = taxable * adj["corporate_tax_rate"]
        ni  = ebt - wht - tax

        df.at[t,"ebt"]        = ebt
        df.at[t,"tax"]        = tax
        df.at[t,"net_income"] = ni

        pf  = ni + df.at[t,"depreciation"] - df.at[t,"capex_total"]
        bal = oc + pf
        if bal < 0:
            draw = -bal
            rep  = 0.0
        else:
            rep  = min(bal, od)
            draw = 0.0

        df.at[t,"debt_draw"]  = draw
        df.at[t,"debt_rep"]   = rep
        df.at[t,"debt_close"] = od + draw - rep
        df.at[t,"cash"]       = oc + pf + draw - rep

    return df
