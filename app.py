import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Weather Detection & Prediction",
    page_icon="🌦️",
    layout="wide"
)



# Authentication removed: dashboard is publicly accessible.

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size:42px;
        font-weight:700;
        text-align:center;
        margin-bottom:5px;
    }

    .subtitle {
        text-align:center;
        font-size:18px;
        margin-bottom:30px;
    }

    .section-title {
        font-size:27px;
        font-weight:600;
        margin-top:20px;
        margin-bottom:10px;
    }

    div[data-testid="stMetric"] {
        padding:15px;
        border-radius:12px;
        border:1px solid rgba(128,128,128,0.3);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    detection_model = joblib.load(
        "models/weather_detection_model.pkl"
    )

    temperature_model = joblib.load(
        "models/temperature_prediction_model.pkl"
    )

    return detection_model, temperature_model


detection_model, temperature_model = load_models()


# ============================================================
# LOAD WEATHER DATA
# ============================================================

@st.cache_data
def load_weather_data():

    data = pd.read_csv(
        "data/cleaned_weather.csv"
    )

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    return data


weather_data = load_weather_data()

available_cities = sorted(
    weather_data["City"].unique()
)


# ============================================================
# HEADER
# ============================================================


st.markdown(
    """
    <div class="main-title">
        🌦️ Weather Detection & Temperature Prediction
    </div>

    <div class="subtitle">
        Data Science based weather analysis and temperature prediction
    </div>
    """,
    unsafe_allow_html=True
)



# ============================================================
# WEATHER ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">🌍 Weather Analysis</div>',
    unsafe_allow_html=True
)


mode = st.radio(
    "Choose Input Mode",
    [
        "🤖 Auto Mode",
        "✍️ Manual Mode"
    ],
    horizontal=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "weather_data" not in st.session_state:

    st.session_state["weather_data"] = None


if "fetched_city" not in st.session_state:

    st.session_state["fetched_city"] = None


# ============================================================
# AUTO MODE
# ============================================================

if mode == "🤖 Auto Mode":

    city = st.selectbox(
        "🏙️ Select City",
        available_cities
    )

    if st.button(
        "🌐 Get Live Weather",
        type="primary"
    ):

        try:

            api_key = st.secrets[
                "OPENWEATHER_API_KEY"
            ]

            response = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "q": city,
                    "appid": api_key,
                    "units": "metric"
                },
                timeout=15
            )

            if response.status_code == 200:

                api_data = response.json()

                main_data = api_data["main"]

                wind_data = api_data["wind"]

                clouds_data = api_data["clouds"]

                temperature = main_data.get(
                    "temp",
                    25.0
                )

                humidity = main_data.get(
                    "humidity",
                    70
                )

                pressure = main_data.get(
                    "pressure",
                    1013
                )

                rainfall = 0.0

                if "rain" in api_data:

                    rainfall = api_data[
                        "rain"
                    ].get(
                        "1h",
                        0.0
                    )

                wind_speed = (
                    wind_data.get(
                        "speed",
                        0.0
                    ) * 3.6
                )

                cloud_cover = clouds_data.get(
                    "all",
                    0
                )

                st.session_state[
                    "weather_data"
                ] = {

                    "temperature": temperature,

                    "humidity": humidity,

                    "rainfall": rainfall,

                    "wind_speed": wind_speed,

                    "pressure": pressure,

                    "cloud_cover": cloud_cover
                }

                st.session_state[
                    "fetched_city"
                ] = city

                st.success(
                    f"✅ Live weather fetched for {city}"
                )

            elif response.status_code == 401:

                st.error(
                    "❌ OpenWeather API key is invalid."
                )

            elif response.status_code == 404:

                st.error(
                    f"❌ Weather data not found for {city}."
                )

            else:

                st.error(
                    f"❌ Weather API error: "
                    f"{response.status_code}"
                )

        except requests.exceptions.Timeout:

            st.error(
                "❌ Weather API request timed out."
            )

        except Exception as e:

            st.error(
                "❌ Could not fetch live weather."
            )

            st.code(str(e))

    if st.session_state[
        "weather_data"
    ] is not None:

        data = st.session_state[
            "weather_data"
        ]

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "🌡️ Temperature",
                f"{data['temperature']:.1f} °C"
            )

        with col2:

            st.metric(
                "💧 Humidity",
                f"{data['humidity']} %"
            )

        with col3:

            st.metric(
                "🌧️ Rainfall",
                f"{data['rainfall']:.2f} mm"
            )

        with col4:

            st.metric(
                "💨 Wind Speed",
                f"{data['wind_speed']:.1f} km/h"
            )

        col5, col6 = st.columns(2)

        with col5:

            st.metric(
                "🔽 Pressure",
                f"{data['pressure']} hPa"
            )

        with col6:

            st.metric(
                "☁️ Cloud Cover",
                f"{data['cloud_cover']} %"
            )


# ============================================================
# MANUAL MODE
# ============================================================

else:

    col1, col2 = st.columns(2)

    with col1:

        city = st.selectbox(
            "🏙️ City",
            available_cities,
            key="manual_city"
        )

        temperature = st.number_input(
            "🌡️ Temperature (°C)",
            value=25.0
        )

        humidity = st.number_input(
            "💧 Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        rainfall = st.number_input(
            "🌧️ Rainfall (mm)",
            min_value=0.0,
            value=0.0
        )

    with col2:

        wind_speed = st.number_input(
            "💨 Wind Speed (km/h)",
            min_value=0.0,
            value=10.0
        )

        pressure = st.number_input(
            "🔽 Pressure (hPa)",
            min_value=800.0,
            max_value=1200.0,
            value=1013.0
        )

        cloud_cover = st.slider(
            "☁️ Cloud Cover (%)",
            0,
            100,
            50
        )

        st.date_input(
            "📅 Date",
            key="manual_date"
        )

    st.session_state[
        "weather_data"
    ] = {

        "temperature": temperature,

        "humidity": humidity,

        "rainfall": rainfall,

        "wind_speed": wind_speed,

        "pressure": pressure,

        "cloud_cover": cloud_cover
    }


# ============================================================
# PREDICTION SECTION
# ============================================================

if st.session_state[
    "weather_data"
] is not None:

    st.divider()

    st.markdown(
        '<div class="section-title">🤖 Weather Prediction</div>',
        unsafe_allow_html=True
    )

    data = st.session_state[
        "weather_data"
    ]

    prediction_date = st.date_input(
        "📅 Prediction Date",
        value=pd.Timestamp.today()
    )

    year = prediction_date.year

    month = prediction_date.month

    day = prediction_date.day

    day_of_year = (
        prediction_date
        .timetuple()
        .tm_yday
    )

    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )

    day_of_year_sin = np.sin(
        2 * np.pi * day_of_year / 365
    )

    day_of_year_cos = np.cos(
        2 * np.pi * day_of_year / 365
    )

    predict_col1, predict_col2 = st.columns(2)

    # ========================================================
    # WEATHER DETECTION
    # ========================================================

    with predict_col1:

        st.markdown(
            "### 🌦️ Weather Condition"
        )

        if st.button(
            "🔍 Detect Weather",
            type="primary",
            width="stretch"
        ):

            try:

                detection_input = pd.DataFrame(

                    [[
                        data["temperature"],
                        data["temperature"],
                        data["temperature"],
                        data["humidity"],
                        data["rainfall"],
                        data["wind_speed"],
                        data["pressure"],
                        data["cloud_cover"]
                    ]],

                    columns=[

                        "Temperature_Max (°C)",

                        "Temperature_Min (°C)",

                        "Temperature_Avg (°C)",

                        "Humidity (%)",

                        "Rainfall (mm)",

                        "Wind_Speed (km/h)",

                        "Pressure (hPa)",

                        "Cloud_Cover (%)"
                    ]
                )

                result = (
                    detection_model
                    .predict(
                        detection_input
                    )[0]
                )

                st.success(
                    f"🌤️ Detected Weather: "
                    f"**{result}**"
                )

            except Exception as e:

                st.error(
                    "❌ Weather detection failed."
                )

                st.code(str(e))

    # ========================================================
    # TEMPERATURE PREDICTION
    # ========================================================

    with predict_col2:

        st.markdown(
            "### 🌡️ Temperature Prediction"
        )

        if st.button(
            "🔮 Predict Temperature",
            type="primary",
            width="stretch"
        ):

            try:

                temperature_input = pd.DataFrame(

                    [[
                        data["humidity"],
                        data["rainfall"],
                        data["wind_speed"],
                        data["pressure"],
                        data["cloud_cover"],
                        year,
                        month,
                        day,
                        day_of_year,
                        month_sin,
                        month_cos,
                        day_of_year_sin,
                        day_of_year_cos,
                        city
                    ]],

                    columns=[

                        "Humidity (%)",

                        "Rainfall (mm)",

                        "Wind_Speed (km/h)",

                        "Pressure (hPa)",

                        "Cloud_Cover (%)",

                        "Year",

                        "Month",

                        "Day",

                        "DayOfYear",

                        "Month_Sin",

                        "Month_Cos",

                        "DayOfYear_Sin",

                        "DayOfYear_Cos",

                        "City"
                    ]
                )

                predicted_temperature = (
                    temperature_model
                    .predict(
                        temperature_input
                    )[0]
                )

                st.success(
                    f"🌡️ Predicted Temperature: "
                    f"**{predicted_temperature:.2f} °C**"
                )

            except Exception as e:

                st.error(
                    "❌ Temperature prediction failed."
                )

                st.code(str(e))


# ============================================================
# HISTORICAL WEATHER DATA
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📊 Historical Weather Data</div>',
    unsafe_allow_html=True
)

chart_city = st.selectbox(
    "🏙️ Select City for Historical Data",
    available_cities,
    key="historical_city"
)

city_data = weather_data[
    weather_data["City"] == chart_city
].copy()


if not city_data.empty:

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.markdown(
            "#### 🌡️ Temperature Trend"
        )

        st.line_chart(
            city_data.set_index("Date")[
                [
                    "Temperature_Max (°C)",
                    "Temperature_Min (°C)",
                    "Temperature_Avg (°C)"
                ]
            ]
        )

    with chart_col2:

        st.markdown(
            "#### 💧 Humidity & Rainfall"
        )

        st.line_chart(
            city_data.set_index("Date")[
                [
                    "Humidity (%)",
                    "Rainfall (mm)"
                ]
            ]
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📈 Model Performance</div>',
    unsafe_allow_html=True
)

performance_col1, performance_col2 = st.columns(2)


with performance_col1:

    st.markdown(
        "### 🌦️ Weather Detection Model"
    )

    st.metric(
        "Accuracy",
        "100%"
    )

    st.info(
        "Random Forest classifier trained "
        "to classify weather conditions."
    )


with performance_col2:

    st.markdown(
        "### 🌡️ Temperature Prediction Model"
    )

    st.metric(
        "MAE",
        "5.07 °C"
    )

    st.metric(
        "RMSE",
        "5.94 °C"
    )

    st.metric(
        "R² Score",
        "0.00"
    )

    st.info(
        "Linear Regression model selected "
        "after comparing multiple algorithms."
    )



# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align: center;
        color: gray;
        font-size: 14px;
        padding: 15px;
        line-height: 2;
    ">
        🌦️ Weather Detection & Temperature Prediction
        <br>
        📊 Data Science
        &nbsp; • &nbsp;
        🌐 OpenWeather API
        &nbsp; • &nbsp;
        🔓 Public Dashboard
    </div>
    """,
    unsafe_allow_html=True
)
