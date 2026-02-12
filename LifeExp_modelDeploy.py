# ========= MODULE IMPORT =========
import os
import numpy as np
import pandas as pd
import streamlit as st
import kagglehub
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip, GeoJsonPopup
import leafmap.foliumap as leafmap
from streamlit_folium import st_folium
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ========= PAGE CONFIG =========
st.set_page_config(
    page_title="Global Life Expectancy Prediction by Country",
    page_icon=':globe_with_meridians:',
    layout="wide",
)

TARGET_COL = "Life expectancy"

SELECTED_FEATURES = [
    "Schooling",
    "Alcohol",
    "infant deaths",
    "BMI",
    "Diphtheria",
    "Polio",
    "percentage expenditure",
    "GDP",
    "Population",
    "Measles",
    "thinness  1-19 years",
    "Hepatitis B",
    "Adult Mortality",
    "under-five deaths",
    "HIV/AIDS",
]

KAGGLE_DATASET = "kumarajarshi/life-expectancy-who"
KAGGLE_FILENAME = "Life Expectancy Data.csv"

# ========= Natural Earth boundaries =========
NE_ADMIN0_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
NE_ADMIN1_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"


# ========= DATA HELPERS =========
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def get_required_input_columns(model) -> list[str] | None:
        """
        Try to infer the raw input columns the pipeline was trained on.
        Works for many sklearn Pipelines/ColumnTransformers.
        """
        # Newer sklearn sometimes stores this directly
        cols = getattr(model, "feature_names_in_", None)
        if cols is not None:
            return list(cols)
        # Try to find a ColumnTransformer inside a Pipeline
        if isinstance(model, Pipeline):
            for step_name, step_obj in model.named_steps.items():
                if isinstance(step_obj, ColumnTransformer):
                    cols = getattr(step_obj, "feature_names_in_", None)
                    if cols is not None:
                        return list(cols)
        return None

def build_aligned_row(required_cols: list[str], inputs: dict, latest: pd.Series, year: int) -> pd.DataFrame:
    row = {}
    for c in required_cols:
        # Priority 1: user input (if present)
        if c in inputs:
            row[c] = inputs[c]
            continue

        # Priority 2: special fields
        if c == "Year":
            row[c] = int(year)
            continue
        if c == "Country":
            # some pipelines include Country; if you have it, fill it; otherwise dummy
            row[c] = str(latest.get("Country", ""))
            continue
        if c == "Status":
            # fill from latest if exists; otherwise a reasonable default
            v = latest.get("Status", "Developing")
            row[c] = "Developing" if pd.isna(v) or v is None or str(v).strip() == "" else v
            continue

        # Priority 3: auto-fill from latest dataset row
        v = latest.get(c, np.nan)
        if pd.isna(v):
            # safe numeric default (your pipeline likely imputes anyway)
            row[c] = 0.0
        else:
            row[c] = v
    return pd.DataFrame([row])

@st.cache_data(show_spinner=False)
def load_kaggle_who_dataset() -> pd.DataFrame:
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_path = os.path.join(path, KAGGLE_FILENAME)
    df = pd.read_csv(csv_path)
    return normalize_columns(df)


# @st.cache_resource(show_spinner=False)
# def train_lasso_model(df: pd.DataFrame) -> Pipeline:
#     required = ["Country", "Year", TARGET_COL] + SELECTED_FEATURES
#     missing = [c for c in required if c not in df.columns]
#     if missing:
#         raise ValueError(f"Missing required columns: {missing}")

#     df2 = df.dropna(subset=["Country", "Year", TARGET_COL]).copy()
#     df2["Country"] = df2["Country"].astype(str)

#     df2 = coerce_numeric(df2, ["Year"] + SELECTED_FEATURES + [TARGET_COL])
#     df2["Year"] = df2["Year"].astype(int)

#     X = df2[SELECTED_FEATURES]
#     y = df2[TARGET_COL].astype(float)
#     groups = df2["Country"]

#     gkf = GroupKFold(n_splits=5)

#     model = Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="median")),
#             (
#                 "lasso",
#                 LassoCV(
#                     alphas=np.logspace(-4, 1, 200),
#                     cv=gkf.split(X, y, groups=groups),
#                     max_iter=50000,
#                     n_jobs=None,
#                     random_state=42,
#                 ),
#             ),
#         ]
#     )
#     model.fit(X, y)
#     return model

def find_actual_row(df: pd.DataFrame, country: str, year: int) -> pd.DataFrame:
    return df[(df["Country"].astype(str) == str(country)) & (df["Year"].astype(int) == int(year))].copy()


def get_latest_country_row(df: pd.DataFrame, country: str) -> pd.Series | None:
    d = df[df["Country"].astype(str) == str(country)].copy()
    if d.empty:
        return None
    d = coerce_numeric(d, ["Year"] + SELECTED_FEATURES + [TARGET_COL])
    d = d.dropna(subset=["Year"]).sort_values("Year")
    if d.empty:
        return None
    return d.iloc[-1]


def metric_years(label: str, value: float):
    st.metric(label=label, value=f"{value:.2f} years")


# ========= MAP HELPERS (Natural Earth Admin-0 + Admin-1 with CLEAN tooltips) =========
@st.cache_data(show_spinner=False)
def load_admin0() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(NE_ADMIN0_URL)
    gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")
    return gdf


@st.cache_data(show_spinner=False)
def load_admin1() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(NE_ADMIN1_URL)
    gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")
    return gdf


def simplify_subset(gdf: gpd.GeoDataFrame, tolerance: float) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty or tolerance <= 0:
        return gdf
    out = gdf.copy()
    out["geometry"] = out["geometry"].simplify(tolerance, preserve_topology=True)
    return out


def add_geojson_with_info(
    fmap: folium.Map,
    gdf: gpd.GeoDataFrame,
    layer_name: str,
    tooltip_fields: list[str],
    tooltip_aliases: list[str] | None = None,
    popup_fields: list[str] | None = None,
    popup_aliases: list[str] | None = None,
    zoom_to: bool = False,
):
    if gdf is None or gdf.empty:
        return

    tooltip_fields = [f for f in tooltip_fields if f in gdf.columns]
    popup_fields = [f for f in (popup_fields or []) if f in gdf.columns]

    style_fn = lambda _: {"weight": 3, "fillOpacity": 0.08}
    highlight_fn = lambda _: {"weight": 5, "fillOpacity": 0.20}

    gj = folium.GeoJson(
        data=gdf.__geo_interface__,
        name=layer_name,
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases if tooltip_aliases else tooltip_fields,
            sticky=False,
            labels=True,
            localize=True,
            max_width=320,
        ),
        popup=GeoJsonPopup(
            fields=popup_fields,
            aliases=popup_aliases if popup_aliases else popup_fields,
            labels=True,
            localize=True,
            max_width=450,
        )
        if popup_fields
        else None,
    )
    gj.add_to(fmap)

    if zoom_to:
        b = gdf.total_bounds  # (minx, miny, maxx, maxy)
        fmap.fit_bounds([[b[1], b[0]], [b[3], b[2]]])


def build_world_map() -> leafmap.Map:
    m = leafmap.Map()
    m.add_basemap("OpenStreetMap")
    return m

# ========= UI =========
st.title("🌍 Global Life Expectancy Prediction by Country")

with st.sidebar:
    st.header("Dataset")
    st.write("The dataset is automatically downloaded from KaggleHub.")

    # Kaggle secrets (won't crash if secrets.toml missing)
    try:
        kaggle_user = st.secrets.get("KAGGLE_USERNAME", None)
        kaggle_key = st.secrets.get("KAGGLE_KEY", None)
        if kaggle_user and kaggle_key:
            os.environ["KAGGLE_USERNAME"] = kaggle_user
            os.environ["KAGGLE_KEY"] = kaggle_key
    except Exception:
        pass

    st.divider()
    st.header("How to use:")
    st.caption(
        "1) Select **Country** (and optional **Province**) to view boundaries.\n"
        "2) Choose **Year**.\n"
        "3) If year is **2000-2015**, we show actual life expectancy (if available).\n"
        "4) If year is **2016+**, fill indicators to predict life expectancy."
    )

# ========= Load dataset =========
try:
    df = load_kaggle_who_dataset()
    data_source = f"KaggleHub: {KAGGLE_DATASET}"
except Exception as e:
    st.error(
        "KaggleHub download failed. This usually means Kaggle credentials are missing.\n\n"
        "Fix options:\n"
        "1) Put kaggle.json in ~/.kaggle/ (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json)\n"
        "2) Set env vars KAGGLE_USERNAME and KAGGLE_KEY\n\n"
        f"Error details: {e}"
    )
    st.stop()

# Normalize + validate
df = normalize_columns(df)

required_core = ["Country", "Year", TARGET_COL]
missing_core = [c for c in required_core if c not in df.columns]
if missing_core:
    st.error(f"Your dataset is missing required columns: {missing_core}")
    st.stop()

missing_feats = [c for c in SELECTED_FEATURES if c not in df.columns]
if missing_feats:
    st.error(f"Your dataset is missing selected feature columns: {missing_feats}")
    st.stop()

df = coerce_numeric(df, ["Year"] + SELECTED_FEATURES + [TARGET_COL])
df = df.dropna(subset=["Country", "Year"]).copy()
df["Country"] = df["Country"].astype(str)
df["Year"] = df["Year"].astype(int)

countries = sorted(df["Country"].unique().tolist())
min_year, max_year = int(df["Year"].min()), int(df["Year"].max())

# credit titles
st.caption(
    f"Dataset downloaded from **{data_source}** | Years: **{min_year}–{max_year}** | Countries: **{len(countries)}**"
)

# Train model (cached)
# with st.spinner("Training Lasso model (cached)…"):
#     model = train_lasso_model(df)
# alpha = float(model.named_steps["lasso"].alpha_)
# st.success(f"Lasso model ready ✅ (alpha = {alpha:.6g})")

# with st.expander("Show Lasso-selected variables (features used)"):
#     st.write(SELECTED_FEATURES)

def add_log_gdp(X):
    """
    Custom transformer used during training.
    Must exist when unpickling the model.
    """
    X = X.copy()
    if "GDP" in X.columns:
        X["GDP_log"] = np.log1p(X["GDP"])
    return X

with open(file='lasso_pipeline.pkl', mode = 'rb') as file:
    model = pickle.load(file=file)

st.success(f"Prediction Model ready ✅")
with st.expander("Indicators used"):
    st.write(SELECTED_FEATURES)

# Load boundaries once (cached)
with st.spinner("Loading Natural Earth boundaries (cached)…"):
    admin0 = load_admin0()
    admin1 = load_admin1()

# Detect fields
ADMIN0_NAME = "ADMIN" if "ADMIN" in admin0.columns else ("NAME" if "NAME" in admin0.columns else None)
ADMIN0_ISO = "ISO_A3" if "ISO_A3" in admin0.columns else ("ADM0_A3" if "ADM0_A3" in admin0.columns else None)

ADMIN1_NAME = "name" if "name" in admin1.columns else ("NAME" if "NAME" in admin1.columns else None)
ADMIN1_LINK = "adm0_a3" if "adm0_a3" in admin1.columns else ("ADM0_A3" if "ADM0_A3" in admin1.columns else None)

if not ADMIN0_NAME or not ADMIN0_ISO or not ADMIN1_NAME or not ADMIN1_LINK:
    st.warning(
        "Boundary field detection failed. The Natural Earth schema might differ. "
        "Try printing admin0.columns/admin1.columns and adjust the column names."
    )

# ========= Controls (world-first) =========
if "ui_country" not in st.session_state:
    st.session_state["ui_country"] = "-- Select a country --"
if "ui_year" not in st.session_state:
    st.session_state["ui_year"] = 2015

country_options = ["-- Select a country --"] + countries

c1, c2 = st.columns([3, 1])

with c1:
    st.selectbox(
        "Choose a country",
        country_options,
        key="ui_country", 
    )

with c2:
    st.number_input(
        "Choose year",
        min_value=2000,
        max_value=2100,
        step=1,
        key="ui_year",      
    )

# Read current values (always up-to-date on rerun)
country = st.session_state["ui_country"]
year = int(st.session_state["ui_year"])

# Map settings (sidebar)
with st.sidebar:
    st.divider()
    st.header("Map settings")
    show_admin1 = st.toggle("Show provinces boundary", value=True)
    province_choice = "(All)"
    simplify_tol = st.slider("Boundary simplify", 0.0, 0.2, 0.0, 0.01)

# ========= MAP =========
st.subheader("🗺️ Map")

# Start with world map by default
m = build_world_map()

if country == "-- Select a country --":
    st.info("World map is shown. Select a country to draw boundaries.")
    st_folium(m, height=720, width=1920, key="map", returned_objects=[])
    st.stop()

# Find Natural Earth country feature by name (exact, then fallback contains)
country_row = admin0[admin0[ADMIN0_NAME].astype(str) == str(country)].copy() if ADMIN0_NAME else admin0.iloc[0:0].copy()

if country_row.empty and ADMIN0_NAME:
    # Fallback: try contains matching
    key = str(country).strip().lower()
    match = admin0[admin0[ADMIN0_NAME].astype(str).str.lower().str.contains(key, na=False)]
    if not match.empty:
        country_row = match.iloc[[0]].copy()

if country_row.empty:
    st.warning("Could not match this country name to the Natural Earth Admin-0 boundary layer.")
    st_folium(m, height=720, width=1920, key="map", returned_objects=[])
    st.stop()

adm0_a3 = str(country_row.iloc[0][ADMIN0_ISO]) if ADMIN0_ISO in country_row.columns else None

# Provinces filtered by adm0_a3
provinces = admin1.iloc[0:0].copy()
if show_admin1 and adm0_a3 and ADMIN1_LINK and ADMIN1_LINK in admin1.columns:
    provinces = admin1[admin1[ADMIN1_LINK].astype(str) == adm0_a3].copy()

    if not provinces.empty:
        prov_list = sorted(provinces[ADMIN1_NAME].dropna().astype(str).unique().tolist())
        with st.sidebar:
            province_choice = st.selectbox("Select a province", ["(All)"] + prov_list, index=0)

# Simplify for speed
country_row = simplify_subset(country_row, simplify_tol)
provinces = simplify_subset(provinces, simplify_tol) if not provinces.empty else provinces

# Add Admin-0 layer with clean tooltip
country_tooltip_fields = [ADMIN0_NAME, ADMIN0_ISO]
country_tooltip_aliases = ["Country: ", "ISO A3: "]
add_geojson_with_info(
    m,
    country_row,
    layer_name=f"{country} (Admin-0)",
    tooltip_fields=country_tooltip_fields,
    tooltip_aliases=country_tooltip_aliases,
    popup_fields=country_tooltip_fields,
    popup_aliases=country_tooltip_aliases,
    zoom_to=True,
)

# Add Admin-1 layer (all or selected) with clean tooltip
if show_admin1 and not provinces.empty:
    if province_choice != "(All)":
        provinces_show = provinces[provinces[ADMIN1_NAME].astype(str) == str(province_choice)].copy()
        layer_name = f"{province_choice}"
        zoom_to = True
    else:
        provinces_show = provinces
        layer_name = f"{country} Provinces"
        zoom_to = False

    tooltip_fields = [ADMIN1_NAME]
    tooltip_aliases = ["Province: "]
    for f, alias in [
        ("adm0_a3", "Country code: "),
        ("gn_a1_code", "GN code: "),
        ("wikidataid", "Wikidata: "),
    ]:
        if f in provinces_show.columns:
            tooltip_fields.append(f)
            tooltip_aliases.append(alias)

    add_geojson_with_info(
        m,
        provinces_show,
        layer_name=layer_name,
        tooltip_fields=tooltip_fields,
        tooltip_aliases=tooltip_aliases,
        popup_fields=tooltip_fields,
        popup_aliases=tooltip_aliases,
        zoom_to=zoom_to,
    )

folium.LayerControl(collapsed=True).add_to(m)

# Critical: prevent rerun loop/popping
st_folium(m, height=720, width=1920, key="map", returned_objects=[])

st.divider()

# ========= If no country selected, stop before analysis/prediction =========
if country == "-- Select a country --":
    st.stop()

# ========= BASELINE: 2000–2015 =========
if 2000 <= year <= 2015:
    st.subheader(f"📌 Life Expectancy Snapshot — Year {year}")
    actual = find_actual_row(df, country, year)

    if not actual.empty and pd.notna(actual.iloc[0][TARGET_COL]):
        le = float(actual.iloc[0][TARGET_COL])
        metric_years(f"Actual life expectancy of **{year}**", le)

        with st.expander("Show record details"):
            show_cols = ["Country", "Year", TARGET_COL] + SELECTED_FEATURES
            st.dataframe(actual[show_cols], use_container_width=True)
    else:
        st.warning(
            "No exact record found for this country-year in 2000-2015. "
            "You can still predict using auto-filled inputs below."
        )
        year = 2016

# ========= PREDICTION: 2016+ =========
if year >= 2016:
    st.subheader(f"🔮 Life Expectancy Prediction — {country.upper()} **({year})**")

    latest = get_latest_country_row(df, country)
    if latest is None:
        st.error("No historical data found for this country to auto-fill predictors.")
        st.stop()

    latest_year = int(latest["Year"])
    st.caption(
        f"Inputs are auto-filled for **{country} from {latest_year}** (latest available). "
        f"Adjust values if you have projections for **{year}**."
    )

    HIGH_POS = [
        "Schooling",
        "Alcohol",
        "infant deaths",
        "BMI",
    ]

    HIGH_NEG = [
        "HIV/AIDS",
        "under-five deaths",
        "Adult Mortality",
        "Hepatitis B",
    ]

    LESS_IMPACT = [
        "Diphtheria",
        "Polio",
        "percentage expenditure",
        "GDP",
        "Population",
        "Measles",
        "thinness  1-19 years",
    ]

    # ========= Render grouped form =========
    with st.form("predict_form"):
        inputs = {}

        def feature_input(col, feat_name: str) -> float:
            default = latest.get(feat_name, np.nan)
            if pd.isna(default):
                default = 0.0

            step = 0.1
            if feat_name in ["Population", "Measles", "under-five deaths", "infant deaths"]:
                step = 1.0

            return float(col.number_input(feat_name, value=float(default), step=float(step)))

        # ========= High Positive =========
        st.markdown("##### 🟢 High Positive Impact")
        c1, c2, c3, c4 = st.columns(4)
        cols = [c1, c2, c3, c4]
        for i, feat in enumerate(HIGH_POS):
            inputs[feat] = feature_input(cols[i % 4], feat)

        st.divider()

        # ========= High Negative =========
        st.markdown("##### 🔴 High Negative Impact")
        c1, c2, c3, c4 = st.columns(4)
        cols = [c1, c2, c3, c4]
        for i, feat in enumerate(HIGH_NEG):
            inputs[feat] = feature_input(cols[i % 4], feat)

        st.divider()

        # ========= Less Impact =========
        st.markdown("##### ⚪ Less Impact")
        c1, c2, c3, c4 = st.columns(4)
        cols = [c1, c2, c3, c4]
        for i, feat in enumerate(LESS_IMPACT):
            inputs[feat] = feature_input(cols[i % 4], feat)

        submitted = st.form_submit_button("Predict life expectancy",type='primary',icon_position='right')

    if submitted:
        # What columns does the loaded model want?
        required_cols = get_required_input_columns(model)

        if not required_cols:
            # Fallback: assume model expects exactly SELECTED_FEATURES
            X_new = pd.DataFrame([inputs], columns=SELECTED_FEATURES)
        else:
            # Build a 1-row dataframe with ALL required columns
            X_new = build_aligned_row(required_cols, inputs, latest, year)

        pred = float(model.predict(X_new)[0])

        cA, cB = st.columns([1, 2])
        with cA:
            st.metric(label=f"Predicted life expectancy ({year})", value=f"{pred:.2f} years")
        with cB:
            st.info(
                "Note: This value is a model prediction based on the indicator inputs you entered. "
                "If some required fields were not provided, they were auto-filled using the latest "
                "available country record or safe defaults."
            )

        with st.expander("Show prediction inputs sent to model"):
            st.dataframe(X_new, use_container_width=True)
