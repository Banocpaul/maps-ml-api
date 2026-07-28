from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# PATHS AND CONSTANTS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"

MANDALUYONG_LATITUDE = 14.5794
MANDALUYONG_LONGITUDE = 121.0359
MANILA_TIMEZONE = "Asia/Manila"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="M.A.P.S. ML API",
    version="1.1.0",
    description=(
        "Automatic live-weather flood classification "
        "and regression service for Mandaluyong City."
    ),
)


# ============================================================
# LOAD MODELS
# ============================================================

def load_model(filename: str) -> Any:
    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise RuntimeError(
            f"Required model file not found: {model_path}"
        )

    return joblib.load(model_path)


classifier = load_model(
    "maps_flood_risk_classifier.joblib"
)

regression_model = load_model(
    "maps_flood_regression_model.joblib"
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class BarangayProfile(BaseModel):
    barangay_id: int
    barangay: str
    nearest_waterway: str

    elevation_m: float
    distance_to_waterway_m: float
    drainage_index: float
    impervious_surface_ratio: float
    population_density_per_km2: float
    historical_flood_count_5y: int


class CitywidePredictionRequest(BaseModel):
    barangays: list[BarangayProfile]


# ============================================================
# HELPERS
# ============================================================

RISK_SCORE = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
}


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        if value is None:
            return default

        return float(value)
    except (TypeError, ValueError):
        return default


def sum_values(
    values: list[Any],
) -> float:
    return round(
        sum(
            safe_float(value)
            for value in values
        ),
        2,
    )


def get_weather_description(
    weather_code: int,
) -> dict[str, str]:
    if weather_code == 0:
        return {
            "condition": "Clear sky",
            "icon": "☀️",
        }

    if weather_code in [1, 2]:
        return {
            "condition": "Partly cloudy",
            "icon": "🌤️",
        }

    if weather_code == 3:
        return {
            "condition": "Overcast",
            "icon": "☁️",
        }

    if weather_code in [45, 48]:
        return {
            "condition": "Foggy",
            "icon": "🌫️",
        }

    if weather_code in [51, 53, 55, 56, 57]:
        return {
            "condition": "Drizzle",
            "icon": "🌦️",
        }

    if weather_code in [61, 63, 65, 66, 67]:
        return {
            "condition": "Rain",
            "icon": "🌧️",
        }

    if weather_code in [80, 81, 82]:
        return {
            "condition": "Rain showers",
            "icon": "🌦️",
        }

    if weather_code in [95, 96, 99]:
        return {
            "condition": "Thunderstorm",
            "icon": "⛈️",
        }

    return {
        "condition": "Unknown weather",
        "icon": "🌡️",
    }


def extract_regression_outputs(
    raw_prediction: Any,
) -> tuple[float, float]:
    values = np.asarray(
        raw_prediction,
        dtype=float,
    ).reshape(-1)

    if len(values) < 2:
        raise RuntimeError(
            "Regression model must return flood depth "
            "and duration."
        )

    predicted_depth = max(
        float(values[0]),
        0.0,
    )

    predicted_duration = max(
        float(values[1]),
        0.0,
    )

    return (
        round(predicted_depth, 2),
        round(predicted_duration, 2),
    )


def get_class_probability(
    model: Any,
    dataframe: pd.DataFrame,
    predicted_class: str,
) -> tuple[float, dict[str, float]]:
    if not hasattr(model, "predict_proba"):
        return 0.0, {}

    probabilities = model.predict_proba(
        dataframe
    )[0]

    classes = [
        str(value)
        for value in model.classes_
    ]

    probability_map = {
        class_name: round(
            float(probability),
            6,
        )
        for class_name, probability
        in zip(classes, probabilities)
    }

    return (
        probability_map.get(
            predicted_class,
            0.0,
        ),
        probability_map,
    )


# ============================================================
# LIVE WEATHER
# ============================================================

async def fetch_live_weather() -> dict[str, Any]:
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": MANDALUYONG_LATITUDE,
        "longitude": MANDALUYONG_LONGITUDE,

        "current": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
            ]
        ),

        "hourly": ",".join(
            [
                "precipitation",
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "weather_code",
            ]
        ),

        "past_days": 7,
        "forecast_days": 7,
        "timezone": MANILA_TIMEZONE,
    }

    try:
        async with httpx.AsyncClient(
            timeout=30.0
        ) as client:
            response = await client.get(
                url,
                params=params,
            )

            response.raise_for_status()

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Weather API request timed out.",
        ) from exc

    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to retrieve weather data: "
                f"{exc}"
            ),
        ) from exc

    payload = response.json()

    current = payload.get("current", {})
    hourly = payload.get("hourly", {})

    times = hourly.get("time", [])
    precipitation = hourly.get(
        "precipitation",
        [],
    )

    if not times or not precipitation:
        raise HTTPException(
            status_code=502,
            detail="Weather API returned incomplete data.",
        )

    manila_now = datetime.now(
        ZoneInfo(MANILA_TIMEZONE)
    )

    naive_now = manila_now.replace(
        tzinfo=None,
        minute=0,
        second=0,
        microsecond=0,
    )

    parsed_times = [
        datetime.fromisoformat(value)
        for value in times
    ]

    current_index = min(
        range(len(parsed_times)),
        key=lambda index: abs(
            parsed_times[index] - naive_now
        ),
    )

    previous_values = precipitation[
        : current_index + 1
    ]

    future_values = precipitation[
        current_index + 1:
    ]

    observed_24h = sum_values(
        previous_values[-24:]
    )

    observed_3d = sum_values(
        previous_values[-72:]
    )

    observed_7d = sum_values(
        previous_values[-168:]
    )

    forecast_24h = sum_values(
        future_values[:24]
    )

    forecast_3d = sum_values(
        future_values[:72]
    )

    forecast_7d = sum_values(
        future_values[:168]
    )

    weather_code = int(
        safe_float(
            current.get("weather_code"),
            0,
        )
    )

    weather_description = get_weather_description(
        weather_code
    )

    return {
        "source": "Open-Meteo",

        "current_date": manila_now.strftime(
            "%B %d, %Y"
        ),

        "current_time": manila_now.strftime(
            "%I:%M:%S %p"
        ),

        "generated_at": manila_now.isoformat(),

        "forecast_horizon": "Next 24 hours",

        "condition":
            weather_description["condition"],

        "weather_icon":
            weather_description["icon"],

        "weather_code":
            weather_code,

        "temperature_c": round(
            safe_float(
                current.get("temperature_2m")
            ),
            2,
        ),

        "humidity_pct": round(
            safe_float(
                current.get(
                    "relative_humidity_2m"
                )
            ),
            2,
        ),

        "wind_speed_kph": round(
            safe_float(
                current.get("wind_speed_10m")
            ),
            2,
        ),

        "current_precipitation_mm": round(
            safe_float(
                current.get("precipitation")
            ),
            2,
        ),

        "observed_rainfall": {
            "past_24h_mm": observed_24h,
            "past_3d_mm": observed_3d,
            "past_7d_mm": observed_7d,
        },

        "forecast_rainfall": {
            "next_24h_mm": forecast_24h,
            "next_3d_mm": forecast_3d,
            "next_7d_mm": forecast_7d,
        },

        # These values are temporary until official automatic
        # PAGASA and tide sources are connected.
        "storm_signal": 0,
        "storm_signal_source":
            "No automatic official PAGASA signal connected",

        "tide_level_m": 0.0,
        "tide_source":
            "No automatic tide source connected",
    }


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home() -> dict[str, Any]:
    return {
        "success": True,
        "message": "M.A.P.S. ML API is running.",
        "version": "1.1.1",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "live_weather": "/weather/live",
            "citywide_prediction": "/predict/citywide",
            "documentation": "/docs",
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """
    Health endpoint used by the Laravel application.

    The deployed API uses one classification model and one combined
    regression model. The combined regression model supplies both
    flood-depth and flood-duration predictions.
    """
    classifier_loaded = classifier is not None
    regression_loaded = regression_model is not None

    return {
        "status": (
            "healthy"
            if classifier_loaded and regression_loaded
            else "degraded"
        ),
        "risk_model_loaded": classifier_loaded,
        "depth_model_loaded": regression_loaded,
        "duration_model_loaded": regression_loaded,
        "metadata_loaded": False,
    }


@app.get("/weather/live")
async def live_weather() -> dict[str, Any]:
    weather = await fetch_live_weather()

    return {
        "success": True,
        "data": weather,
    }


@app.post("/predict/citywide")
async def predict_citywide(
    request: CitywidePredictionRequest,
) -> dict[str, Any]:
    if not request.barangays:
        raise HTTPException(
            status_code=422,
            detail="No barangay profiles supplied.",
        )

    weather = await fetch_live_weather()

    manila_now = datetime.now(
        ZoneInfo(MANILA_TIMEZONE)
    )

    month = manila_now.month

    is_weekend = (
        1
        if manila_now.weekday() >= 5
        else 0
    )

    wet_season = (
        1
        if 5 <= month <= 11
        else 0
    )

    storm_signal = int(
        weather["storm_signal"]
    )

    tide_level_m = float(
        weather["tide_level_m"]
    )

    observed = weather[
        "observed_rainfall"
    ]

    forecast = weather[
        "forecast_rainfall"
    ]

    results: list[dict[str, Any]] = []

    for profile in request.barangays:
        classification_input = pd.DataFrame(
            [
                {
                    "month": month,
                    "is_weekend": is_weekend,
                    "wet_season": wet_season,
                    "storm_signal": storm_signal,
                    "barangay":
                        profile.barangay,
                    "nearest_waterway":
                        profile.nearest_waterway,
                    "elevation_m":
                        profile.elevation_m,
                    "distance_to_waterway_m":
                        profile.distance_to_waterway_m,
                    "drainage_index":
                        profile.drainage_index,
                    "impervious_surface_ratio":
                        profile.impervious_surface_ratio,
                    "population_density_per_km2":
                        profile.population_density_per_km2,
                    "historical_flood_count_5y":
                        profile.historical_flood_count_5y,

                    "rainfall_24h_mm":
                        forecast["next_24h_mm"],

                    "rainfall_3d_mm":
                        observed["past_3d_mm"]
                        + forecast["next_3d_mm"],

                    "rainfall_7d_mm":
                        observed["past_7d_mm"]
                        + forecast["next_7d_mm"],

                    "temperature_c":
                        weather["temperature_c"],

                    "humidity_pct":
                        weather["humidity_pct"],

                    "wind_speed_kph":
                        weather["wind_speed_kph"],

                    "tide_level_m":
                        tide_level_m,
                }
            ]
        )

        regression_input = pd.DataFrame(
            [
                {
                    "month": month,
                    "wet_season": wet_season,
                    "storm_signal": storm_signal,
                    "barangay":
                        profile.barangay,
                    "nearest_waterway":
                        profile.nearest_waterway,
                    "elevation_m":
                        profile.elevation_m,
                    "distance_to_waterway_m":
                        profile.distance_to_waterway_m,
                    "drainage_index":
                        profile.drainage_index,
                    "impervious_surface_ratio":
                        profile.impervious_surface_ratio,
                    "population_density_per_km2":
                        profile.population_density_per_km2,
                    "historical_flood_count_5y":
                        profile.historical_flood_count_5y,

                    "rainfall_24h_mm":
                        forecast["next_24h_mm"],

                    "rainfall_3d_mm":
                        observed["past_3d_mm"]
                        + forecast["next_3d_mm"],

                    "rainfall_7d_mm":
                        observed["past_7d_mm"]
                        + forecast["next_7d_mm"],

                    "temperature_c":
                        weather["temperature_c"],

                    "humidity_pct":
                        weather["humidity_pct"],

                    "wind_speed_kph":
                        weather["wind_speed_kph"],

                    "tide_level_m":
                        tide_level_m,
                }
            ]
        )

        try:
            risk_level = str(
                classifier.predict(
                    classification_input
                )[0]
            )

            confidence, probabilities = (
                get_class_probability(
                    classifier,
                    classification_input,
                    risk_level,
                )
            )

            raw_regression = (
                regression_model.predict(
                    regression_input
                )[0]
            )

            (
                predicted_depth,
                predicted_duration,
            ) = extract_regression_outputs(
                raw_regression
            )

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Prediction failed for barangay "
                    f"'{profile.barangay}': {exc}"
                ),
            ) from exc

        results.append(
            {
                "barangay_id":
                    profile.barangay_id,

                "barangay":
                    profile.barangay,

                "risk_level":
                    risk_level,

                "risk_score":
                    RISK_SCORE.get(
                        risk_level,
                        0,
                    ),

                "confidence":
                    round(
                        confidence,
                        6,
                    ),

                "probabilities":
                    probabilities,

                "predicted_depth_mm":
                    predicted_depth,

                "predicted_duration_hours":
                    predicted_duration,

                "cluster_number":
                    "Not used for live prediction",

                "profile": {
                    "nearest_waterway":
                        profile.nearest_waterway,

                    "elevation_m":
                        profile.elevation_m,

                    "distance_to_waterway_m":
                        profile.distance_to_waterway_m,

                    "drainage_index":
                        profile.drainage_index,

                    "historical_flood_count_5y":
                        profile.historical_flood_count_5y,
                },
            }
        )

    results.sort(
        key=lambda item: (
            item["risk_score"],
            item["confidence"],
            item["predicted_depth_mm"],
        ),
        reverse=True,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        result["rank"] = rank

    risk_distribution = {
        "Low": sum(
            1
            for result in results
            if result["risk_level"] == "Low"
        ),
        "Medium": sum(
            1
            for result in results
            if result["risk_level"] == "Medium"
        ),
        "High": sum(
            1
            for result in results
            if result["risk_level"] == "High"
        ),
    }

    return {
        "success": True,

        "generated_at":
            manila_now.isoformat(),

        "prediction_horizon":
            "Next 24 hours",

        "weather":
            weather,

        "summary": {
            "total_barangays":
                len(results),

            "risk_distribution":
                risk_distribution,

            "highest_predicted_depth_mm":
                max(
                    item["predicted_depth_mm"]
                    for item in results
                ),

            "longest_predicted_duration_hours":
                max(
                    item[
                        "predicted_duration_hours"
                    ]
                    for item in results
                ),
        },

        "predictions":
            results,
    }