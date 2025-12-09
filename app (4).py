import os
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import streamlit as st
from opencage.geocoder import OpenCageGeocode

# ---------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Cotton Heat Unit Planner (DD60 / DD55)",
    layout="wide",
)

# Updated DD60 requirements from Raper et al. (2023) – approximate means
COTTON_DD60_THRESHOLDS = {
    "Emergence": 106,
    "First Square": 670,
    "First Flower": 1116,
    "Cutout": 1618,
    "First Cracked Boll": 2209,
    "60% Open Bolls": 2523,
}


# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def geocode_place(place_name: str):
    """
    Geocode a place name using OpenCage.
    Requires OPENCAGE_API_KEY in environment or st.secrets["OPENCAGE_API_KEY"].
    """
    api_key = os.environ.get("OPENCAGE_API_KEY") or st.secrets.get("OPENCAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENCAGE_API_KEY not set. "
            "Add it to your environment variables or Streamlit secrets."
        )

    geocoder = OpenCageGeocode(api_key)
    results = geocoder.geocode(place_name, limit=1, no_annotations=1)

    if not results:
        raise ValueError(f"Place not found: {place_name}")

    best = results[0]
    lat = best["geometry"]["lat"]
    lon = best["geometry"]["lng"]
    formatted = best.get("formatted", place_name)
    return lat, lon, formatted


def fetch_nasa_power_daily_tmax_tmin(lat: float, lon: float,
                                     start_date: date,
                                     end_date: date) -> pd.DataFrame:
    """
    Fetch daily Tmax/Tmin from NASA POWER AG community API.

    Parameters returned:
      - T2M_MAX: daily max air temperature at 2 m (°C)
      - T2M_MIN: daily min air temperature at 2 m (°C)

    We convert °C → °F for GDD calculations.
    """

    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M_MAX,T2M_MIN&community=AG"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start_str}&end={end_str}&format=JSON"
    )

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # POWER daily data are usually under:
    # data["properties"]["parameter"]["T2M_MAX"][date_str]
    params = data.get("properties", {}).get("parameter", {})
    tmax_dict = params.get("T2M_MAX", {})
    tmin_dict = params.get("T2M_MIN", {})

    if not tmax_dict or not tmin_dict:
        raise ValueError("NASA POWER response missing T2M_MAX/T2M_MIN data.")

    records = []
    for d_str in sorted(tmax_dict.keys()):
        tmax_c = tmax_dict[d_str]
        tmin_c = tmin_dict.get(d_str)
        if tmin_c is None:
            continue

        # Convert °C to °F
        tmax_f = tmax_c * 9 / 5 + 32
        tmin_f = tmin_c * 9 / 5 + 32

        records.append(
            {
                "date": datetime.strptime(d_str, "%Y%m%d").date(),
                "tmax_F": tmax_f,
                "tmin_F": tmin_f,
            }
        )

    df = pd.DataFrame.from_records(records).sort_values("date")
    return df


def gdd_dd60(tmax_f: float, tmin_f: float) -> float:
    """Standard DD60: base 60°F, no upper cap."""
    t_avg = (tmax_f + tmin_f) / 2
    return max(t_avg - 60.0, 0.0)


def gdd_dd55(tmax_f: float, tmin_f: float) -> float:
    """DD55: base 55°F, no upper cap."""
    t_avg = (tmax_f + tmin_f) / 2
    return max(t_avg - 55.0, 0.0)


def gdd_dd55_upper86(tmax_f: float, tmin_f: float) -> float:
    """DD55 with Tmax capped at 86°F (no development above 86°F)."""
    tmax_adj = min(tmax_f, 86.0)
    t_avg = (tmax_adj + tmin_f) / 2
    return max(t_avg - 55.0, 0.0)


def compute_gdd(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Add daily and cumulative GDD columns to dataframe based on chosen model.
    """
    if model == "DD60 (base 60°F)":
        df["GDD"] = df.apply(lambda r: gdd_dd60(r["tmax_F"], r["tmin_F"]), axis=1)
    elif model == "DD55 (base 55°F)":
        df["GDD"] = df.apply(lambda r: gdd_dd55(r["tmax_F"], r["tmin_F"]), axis=1)
    elif model == "DD55 + 86°F cap":
        df["GDD"] = df.apply(lambda r: gdd_dd55_upper86(r["tmax_F"], r["tmin_F"]), axis=1)
    else:
        raise ValueError(f"Unknown model: {model}")

    df["GDD_cum"] = df["GDD"].cumsum()
    return df


def predict_stage_dates(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Given a cumulative GDD dataframe and DD thresholds, find the first date
    when each stage threshold is reached or exceeded.
    """
    stage_rows = []
    for stage, thr in thresholds.items():
        sub = df[df["GDD_cum"] >= thr]
        if sub.empty:
            stage_date = None
        else:
            stage_date = sub.iloc[0]["date"]
        stage_rows.append({"Stage": stage, "Threshold_DD60": thr, "Predicted_Date": stage_date})
    return pd.DataFrame(stage_rows)


# ---------------------------------------------------------
# 3. UI LAYOUT
# ---------------------------------------------------------

st.title("Cotton Heat Unit Planner (DD60 / DD55 / DD55+86°F)")
with st.sidebar:
    st.header(" Location & Dates")

    place_mode = st.radio(
        "How do you want to specify location?",
        ["Place name (geocoding)", "Latitude / Longitude"],
    )

    if place_mode == "Place name (geocoding)":
        place_name = st.text_input("Place name", value="Quincy, Florida, USA")
        lat = lon = None
    else:
        lat = st.number_input("Latitude (°)", value=30.54, format="%.4f")
        lon = st.number_input("Longitude (°)", value=-84.60, format="%.4f")
        place_name = None

    planting_date = st.date_input(
        "Planting date",
        value=date(date.today().year, 4, 15),
    )

    end_date = st.date_input(
        "End date for GDD calculation",
        value=min(date.today(), planting_date + timedelta(days=180)),
        help="Usually 150–180 days after planting for cotton.",
    )

    st.markdown("---")
    st.header("GDD Model")

    model = st.radio(
        "Select GDD model",
        [
            "DD60 (base 60°F)",
            "DD55 (base 55°F)",
            "DD55 + 86°F cap",
        ],
    )

    st.markdown(
        """
        - **DD60** is the standard US cotton model.  
        - **DD55** and **DD55 + 86°F** are experimental models 
        """
    )

    run_button = st.button(" Run Cotton GDD Analysis")


# ---------------------------------------------------------
# 4. MAIN LOGIC
# ---------------------------------------------------------

if run_button:
    try:
        # Resolve location
        if place_mode == "Place name (geocoding)":
            if not place_name.strip():
                st.error("Please enter a valid place name.")
                st.stop()
            lat, lon, formatted = geocode_place(place_name)
            st.success(f"Location resolved: **{formatted}** (lat: {lat:.4f}, lon: {lon:.4f})")
        else:
            formatted = f"Lat {lat:.4f}, Lon {lon:.4f}"
            st.info(f"Using coordinates: **{formatted}**")

        # Fetch NASA POWER data
        with st.spinner("Fetching daily temperature data from NASA POWER..."):
            df_temp = fetch_nasa_power_daily_tmax_tmin(lat, lon, planting_date, end_date)

        if df_temp.empty:
            st.warning("No temperature data returned for the selected dates/location.")
            st.stop()

        # Compute GDD and cumulative GDD
        df_gdd = compute_gdd(df_temp.copy(), model=model)

        st.subheader(" Daily Temperatures & GDD")
        st.write(f"Location: **{formatted}**")
        st.dataframe(df_gdd, use_container_width=True, hide_index=True)

        # Line chart of cumulative GDD
        st.markdown("#### Cumulative GDD Over Time")
        chart_df = df_gdd[["date", "GDD_cum"]].set_index("date")
        st.line_chart(chart_df)

        # Predict growth stages (using DD60 thresholds as reference)
        st.markdown("### Predicted Cotton Growth Stages (Using DD Thresholds)")

        st.info(
            "When you switch to DD55 / DD55+86°F, this becomes an *experimental* comparison."
        )

        stage_df = predict_stage_dates(df_gdd, COTTON_DD60_THRESHOLDS)
        st.dataframe(stage_df, use_container_width=True, hide_index=True)

        # Nice cards for stages
        cols = st.columns(len(COTTON_DD60_THRESHOLDS))
        for i, row in stage_df.iterrows():
            with cols[i]:
                st.metric(
                    label=row["Stage"],
                    value=row["Predicted_Date"].strftime("%Y-%m-%d") if row["Predicted_Date"] else "Not reached",
                    delta=f"{row['Threshold_DD60']} DD60",
                )

      

    except Exception as e:
        st.error(f"Something went wrong: {e}")

else:
    st.markdown(
        """
        ###  How to use this tool

        1. **Choose a location** in the sidebar (place name or lat/lon).  
        2. **Set planting date** and an end date for the season.  
        3. **Select a GDD model**:
           - DD60: standard US cotton practice.  
           - DD55: experimental lower base temperature.  
           - DD55 + 86°F cap: also limits development above 86°F.  
        4. Click **“Run Cotton GDD Analysis”**.

        The app will then:
        - Pull daily temperature data from **NASA POWER**.  
        - Compute **daily & cumulative GDD**.  
        - Estimate when your crop will hit key stages:
          emergence, first square, first flower, cutout,
          first cracked boll, and 60% open.

        Use this as a **research & learning tool** or a **support tool** alongside
        your local extension recommendations.
        """
    )
