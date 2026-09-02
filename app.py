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


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
            margin-bottom: 30px;
        }

        .section-title {
            font-size: 27px;
            font-weight: 600;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        div[data-testid="stMetric"] {
            padding: 15px;
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, 0.3);
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
    '<div class="main-title">🌦️ Weather Detection & Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning powered weather analysis using Python'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# WEATHER INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">📍 Weather Information</div>',
    unsafe_allow_html=True
)

st.write(
    "Choose Auto Mode for live weather data or Manual Mode "
    "to enter weather conditions yourself."
)


# ============================================================
# MODE SELECTION
# ============================================================

mode = st.radio(
    "⚙️ Select Mode",
    ["🤖 Auto Mode", "✋ Manual Mode"],
    horizontal=True
)


# ============================================================
# DEFAULT VALUES
# ============================================================

city = available_cities[0]

date = pd.Timestamp.today().date()

temperature_max = 30.0
temperature_min = 20.0
humidity = 60.0
rainfall = 0.0
wind_speed = 5.0
pressure = 1010.0
cloud_cover = 50


# ============================================================
# AUTO MODE
# ============================================================

if mode == "🤖 Auto Mode":

    st.info(
        "🤖 Auto Mode fetches the latest weather conditions "
        "from OpenWeather."
    )

    city = st.selectbox(
        "🏙️ Select City",
        available_cities,
        key="auto_city"
    )

    # --------------------------------------------------------
    # CLEAR OLD WEATHER WHEN CITY CHANGES
    # --------------------------------------------------------

    if (
        "fetched_city" in st.session_state
        and st.session_state["fetched_city"] != city
    ):

        st.session_state.pop(
            "weather_data",
            None
        )

        st.session_state.pop(
            "fetched_city",
            None
        )


    # --------------------------------------------------------
    # GET LIVE WEATHER BUTTON
    # --------------------------------------------------------

    if st.button(
        "🌦️ Get Live Weather",
        use_container_width=True
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
                timeout=10
            )

            response.raise_for_status()

            weather = response.json()


            # ------------------------------------------------
            # EXTRACT WEATHER DATA
            # ------------------------------------------------

            temperature = weather["main"]["temp"]

            temperature_max = weather["main"]["temp_max"]

            temperature_min = weather["main"]["temp_min"]

            humidity = weather["main"]["humidity"]

            pressure = weather["main"]["pressure"]

            rainfall = weather.get(
                "rain",
                {}
            ).get(
                "1h",
                0.0
            )

            # OpenWeather gives wind speed in m/s.
            # Our ML model expects km/h.
            wind_speed = (
                weather["wind"]["speed"] * 3.6
            )

            cloud_cover = weather["clouds"]["all"]

            weather_condition = weather[
                "weather"
            ][0]["main"]

            weather_description = weather[
                "weather"
            ][0]["description"]


            # ------------------------------------------------
            # SAVE WEATHER DATA
            # ------------------------------------------------

            st.session_state["weather_data"] = {

                "temperature": temperature,

                "temperature_max": temperature_max,

                "temperature_min": temperature_min,

                "humidity": humidity,

                "rainfall": rainfall,

                "wind_speed": wind_speed,

                "pressure": pressure,

                "cloud_cover": cloud_cover,

                "weather_condition": weather_condition,

                "weather_description": weather_description
            }


            # Remember which city the data belongs to
            st.session_state[
                "fetched_city"
            ] = city


            st.success(
                "✅ Live weather data fetched successfully!"
            )


        except requests.exceptions.HTTPError:

            st.error(
                "❌ OpenWeather rejected the request. "
                "Please check your API key or city."
            )


        except requests.exceptions.RequestException as e:

            st.error(
                f"❌ Unable to connect to OpenWeather: {e}"
            )


        except KeyError:

            st.error(
                "❌ Unexpected weather data received "
                "from OpenWeather."
            )


    # ========================================================
    # DISPLAY LIVE WEATHER
    # ========================================================

    if "weather_data" in st.session_state:

        live_weather = st.session_state[
            "weather_data"
        ]


        # ----------------------------------------------------
        # LOAD SAVED VALUES
        # ----------------------------------------------------

        temperature_max = live_weather[
            "temperature_max"
        ]

        temperature_min = live_weather[
            "temperature_min"
        ]

        humidity = live_weather[
            "humidity"
        ]

        rainfall = live_weather[
            "rainfall"
        ]

        wind_speed = live_weather[
            "wind_speed"
        ]

        pressure = live_weather[
            "pressure"
        ]

        cloud_cover = live_weather[
            "cloud_cover"
        ]


        # ----------------------------------------------------
        # LIVE WEATHER HEADER
        # ----------------------------------------------------

        st.subheader(
            "🌦️ Live Weather Conditions"
        )


        st.success(
            f"🌤️ Current Condition: "
            f"**{live_weather['weather_condition']}** "
            f"— {live_weather['weather_description'].title()}"
        )


        # ----------------------------------------------------
        # MAIN WEATHER METRICS
        # ----------------------------------------------------

        live_col1, live_col2, live_col3, live_col4 = st.columns(4)


        with live_col1:

            st.metric(
                "🌡️ Temperature",
                f"{live_weather['temperature']:.1f} °C"
            )


        with live_col2:

            st.metric(
                "💧 Humidity",
                f"{humidity:.0f} %"
            )


        with live_col3:

            st.metric(
                "🌧️ Rainfall",
                f"{rainfall:.2f} mm"
            )


        with live_col4:

            st.metric(
                "☁️ Cloud Cover",
                f"{cloud_cover:.0f} %"
            )


        st.write("")


        # ----------------------------------------------------
        # ADDITIONAL WEATHER DETAILS
        # ----------------------------------------------------

        detail_col1, detail_col2, detail_col3 = st.columns(3)


        with detail_col1:

            st.metric(
                "💨 Wind Speed",
                f"{wind_speed:.1f} km/h"
            )


        with detail_col2:

            st.metric(
                "🔽 Pressure",
                f"{pressure:.0f} hPa"
            )


        with detail_col3:

            st.metric(
                "🌡️ Max / Min",
                f"{temperature_max:.1f} / "
                f"{temperature_min:.1f} °C"
            )


        date = pd.Timestamp.today().date()


# ============================================================
# MANUAL MODE
# ============================================================

else:

    st.info(
        "✋ Manual Mode allows you to enter weather "
        "conditions yourself."
    )


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

    with col1:

        city = st.selectbox(
            "🏙️ City",
            available_cities,
            key="manual_city"
        )


        temperature_max = st.number_input(
            "🌡️ Maximum Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=30.0,
            step=0.1
        )


        temperature_min = st.number_input(
            "🌡️ Minimum Temperature (°C)",
            min_value=-10.0,
            max_value=50.0,
            value=20.0,
            step=0.1
        )


        humidity = st.number_input(
            "💧 Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=0.1
        )


        rainfall = st.number_input(
            "🌧️ Rainfall (mm)",
            min_value=0.0,
            value=0.0,
            step=0.1
        )


    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with col2:

        date = st.date_input(
            "📅 Date"
        )


        wind_speed = st.number_input(
            "💨 Wind Speed (km/h)",
            min_value=0.0,
            value=5.0,
            step=0.1
        )


        pressure = st.number_input(
            "🌡️ Pressure (hPa)",
            min_value=900.0,
            max_value=1100.0,
            value=1010.0,
            step=0.1
        )


        cloud_cover = st.slider(
            "☁️ Cloud Cover (%)",
            min_value=0,
            max_value=100,
            value=50
        )


# ============================================================
# PREDICTION BUTTON
# ============================================================

prediction_ready = (

    mode == "✋ Manual Mode"

    or (

        "weather_data" in st.session_state

        and st.session_state.get(
            "fetched_city"
        ) == city
    )
)


predict_button = st.button(
    "🔍 Detect Weather & Predict Temperature",
    use_container_width=True,
    disabled=not prediction_ready
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    with st.spinner(
        "Analyzing weather conditions..."
    ):


        # ====================================================
        # DATE FEATURES
        # ====================================================

        year = date.year

        month = date.month

        day = date.day

        day_of_year = date.timetuple().tm_yday


        month_sin = np.sin(
            2 * np.pi * month / 12
        )


        month_cos = np.cos(
            2 * np.pi * month / 12
        )


        day_of_year_sin = np.sin(
            2 * np.pi * day_of_year / 365.25
        )


        day_of_year_cos = np.cos(
            2 * np.pi * day_of_year / 365.25
        )


        # ====================================================
        # DETECTION MODEL INPUT
        # ====================================================

        detection_input = pd.DataFrame(
            {

                "Temperature_Max (°C)": [
                    temperature_max
                ],

                "Temperature_Min (°C)": [
                    temperature_min
                ],

                "Temperature_Avg (°C)": [
                    (
                        temperature_max
                        + temperature_min
                    ) / 2
                ],

                "Humidity (%)": [
                    humidity
                ],

                "Rainfall (mm)": [
                    rainfall
                ],

                "Wind_Speed (km/h)": [
                    wind_speed
                ],

                "Pressure (hPa)": [
                    pressure
                ],

                "Cloud_Cover (%)": [
                    cloud_cover
                ]
            }
        )


        # ====================================================
        # WEATHER DETECTION
        # ====================================================

        detected_weather = detection_model.predict(
            detection_input
        )[0]


        # ====================================================
        # TEMPERATURE MODEL INPUT
        # ====================================================

        temperature_input = pd.DataFrame(
            {

                "Humidity (%)": [
                    humidity
                ],

                "Rainfall (mm)": [
                    rainfall
                ],

                "Wind_Speed (km/h)": [
                    wind_speed
                ],

                "Pressure (hPa)": [
                    pressure
                ],

                "Cloud_Cover (%)": [
                    cloud_cover
                ],

                "Year": [
                    year
                ],

                "Month": [
                    month
                ],

                "Day": [
                    day
                ],

                "DayOfYear": [
                    day_of_year
                ],

                "Month_Sin": [
                    month_sin
                ],

                "Month_Cos": [
                    month_cos
                ],

                "DayOfYear_Sin": [
                    day_of_year_sin
                ],

                "DayOfYear_Cos": [
                    day_of_year_cos
                ],

                "City": [
                    city
                ]
            }
        )


        # ====================================================
        # TEMPERATURE PREDICTION
        # ====================================================

        predicted_temperature = temperature_model.predict(
            temperature_input
        )[0]


    # ========================================================
    # PREDICTION RESULTS
    # ========================================================

    st.divider()


    st.markdown(
        '<div class="section-title">'
        '🎯 Prediction Results'
        '</div>',
        unsafe_allow_html=True
    )


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "🌤️ Detected Weather",
            detected_weather
        )


    with result_col2:

        st.metric(
            "🌡️ Predicted Average Temperature",
            f"{predicted_temperature:.1f} °C"
        )


    st.success(
        "✅ Prediction completed successfully!"
    )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📋 Input Summary'
        '</div>',
        unsafe_allow_html=True
    )


    summary = pd.DataFrame(
        {

            "Parameter": [

                "City",

                "Date",

                "Maximum Temperature",

                "Minimum Temperature",

                "Humidity",

                "Rainfall",

                "Wind Speed",

                "Pressure",

                "Cloud Cover"
            ],


            "Value": [

                city,

                str(date),

                f"{temperature_max:.1f} °C",

                f"{temperature_min:.1f} °C",

                f"{humidity:.1f} %",

                f"{rainfall:.1f} mm",

                f"{wind_speed:.1f} km/h",

                f"{pressure:.1f} hPa",

                f"{cloud_cover:.1f} %"
            ]
        }
    )


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# HISTORICAL WEATHER DATA
# ============================================================

st.divider()


st.markdown(
    '<div class="section-title">'
    '📈 Historical Weather Analysis'
    '</div>',
    unsafe_allow_html=True
)


st.write(
    "Explore historical weather patterns from the dataset."
)


graph_city = st.selectbox(
    "🏙️ Select a city for historical analysis",
    available_cities,
    key="graph_city"
)


city_data = weather_data[
    weather_data["City"] == graph_city
].sort_values(
    "Date"
)


# ============================================================
# TEMPERATURE TREND
# ============================================================

st.subheader(
    f"🌡️ Temperature Trend — {graph_city}"
)


temperature_chart = city_data[
    [
        "Date",
        "Temperature_Avg (°C)"
    ]
].set_index(
    "Date"
)


st.line_chart(
    temperature_chart,
    use_container_width=True
)


# ============================================================
# RAINFALL TREND
# ============================================================

st.subheader(
    f"🌧️ Rainfall Trend — {graph_city}"
)


rainfall_chart = city_data[
    [
        "Date",
        "Rainfall (mm)"
    ]
].set_index(
    "Date"
)


st.line_chart(
    rainfall_chart,
    use_container_width=True
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()


st.markdown(
    '<div class="section-title">'
    '🤖 Model Performance'
    '</div>',
    unsafe_allow_html=True
)


st.write(
    "Performance of the machine learning models evaluated "
    "on the test dataset."
)


# ============================================================
# WEATHER DETECTION MODEL
# ============================================================

st.subheader(
    "🌤️ Weather Detection Model"
)


detection_col1, detection_col2, detection_col3 = st.columns(3)


with detection_col1:

    st.metric(
        "Accuracy",
        "100%"
    )


with detection_col2:

    st.metric(
        "Algorithm",
        "Random Forest"
    )


with detection_col3:

    st.metric(
        "Classes",
        "4"
    )


st.write(
    "The model detects four weather conditions: "
    "Sunny, Cloudy, Humid, and Rainy."
)


# ============================================================
# TEMPERATURE PREDICTION MODEL
# ============================================================

st.subheader(
    "🌡️ Temperature Prediction Model"
)


temperature_col1, temperature_col2, temperature_col3 = st.columns(3)


with temperature_col1:

    st.metric(
        "MAE",
        "5.07 °C"
    )


with temperature_col2:

    st.metric(
        "RMSE",
        "5.94 °C"
    )


with temperature_col3:

    st.metric(
        "R² Score",
        "≈ 0.00"
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader(
    "📊 Temperature Model Comparison"
)


comparison_data = pd.DataFrame(
    {

        "Model": [

            "Linear Regression",

            "Gradient Boosting",

            "Random Forest",

            "Decision Tree"
        ],


        "MAE (°C)": [

            5.07,

            5.12,

            5.12,

            5.94
        ],


        "RMSE (°C)": [

            5.94,

            6.02,

            6.03,

            7.19
        ],


        "R² Score": [

            0.00,

            -0.03,

            -0.03,

            -0.47
        ]
    }
)


st.dataframe(
    comparison_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# BEST MODEL
# ============================================================

st.success(
    "🏆 Best Temperature Model: Linear Regression"
)


# ============================================================
# PERFORMANCE NOTE
# ============================================================

st.info(
    "The weather detection model achieved 100% accuracy "
    "on the test set. Among the tested temperature models, "
    "Linear Regression performed best with the lowest MAE "
    "and RMSE."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Weather Detection & Temperature Prediction | "
    "Data Science Internship Project | "
    "Python • Pandas • Scikit-learn • Streamlit"
)