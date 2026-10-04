from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from datetime import datetime, timedelta
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

ALLOWED_FORECAST_HOURS = {24, 48, 72}
SEVERITY_RANK = {"A": 1, "B": 2, "C": 3, "D": 4}

WEATHER_CACHE_TTL_MINUTES = 30
WEATHER_CACHE_FILE = APP_DIR / "weather_cache.json"
OPEN_METEO_MAX_ATTEMPTS = 3

_weather_cache: dict[str, Any] | None = None
_weather_cache_fetched_at: datetime | None = None
_weather_cache_lock = asyncio.Lock()


# ============================================================
# WEATHER CACHE
# ============================================================

def load_persisted_weather_cache() -> tuple[
    dict[str, Any] | None,
    datetime | None,
]:
    if not WEATHER_CACHE_FILE.exists():
        return None, None

    try:
        persisted = json.loads(
            WEATHER_CACHE_FILE.read_text(encoding="utf-8")
        )
        weather = persisted.get("weather")
        fetched_at_raw = persisted.get("fetched_at")

        if not isinstance(weather, dict) or not fetched_at_raw:
            return None, None

        fetched_at = datetime.fromisoformat(str(fetched_at_raw))
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(
                tzinfo=ZoneInfo(MANILA_TIMEZONE)
            )

        return weather, fetched_at
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None, None


def persist_weather_cache(
    weather: dict[str, Any],
    fetched_at: datetime,
) -> None:
    temporary_file = WEATHER_CACHE_FILE.with_suffix(".tmp")

    try:
        temporary_file.write_text(
            json.dumps(
                {
                    "fetched_at": fetched_at.isoformat(),
                    "weather": weather,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary_file.replace(WEATHER_CACHE_FILE)
    except OSError:
        temporary_file.unlink(missing_ok=True)


_weather_cache, _weather_cache_fetched_at = (
    load_persisted_weather_cache()
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="M.A.P.S. ML API",
    version="2.3.0",
    description=(
        "A-D flood severity prediction for Mandaluyong City with live "
        "24/48/72-hour forecasts and rainfall-only simulation."
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


flood_code_classifier = load_model(
    "maps_flood_code_classifier.joblib"
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

    waterway_type: str = "Unknown"
    previous_floods_30d: int = 0
    days_since_previous_flood: float = 999.0


class RainfallSimulation(BaseModel):
    rainfall_24h_mm: float
    rainfall_3d_mm: float
    rainfall_7d_mm: float


class CitywidePredictionRequest(BaseModel):
    forecast_hours: int = 24
    barangays: list[BarangayProfile]
    simulation: RainfallSimulation | None = None


# ============================================================
# HELPERS
# ============================================================

RISK_SCORE = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
}

FLOOD_SEVERITY_LABELS = {
    "A": "Level A - Minor Flooding (0.5 ft)",
    "B": "Level B - Moderate Flooding (1.5 ft)",
    "C": "Level C - Severe Flooding (2.0 ft)",
    "D": "Level D - Critical Flooding (2.5 ft+)",
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


def sum_values(values: list[Any]) -> float:
    return round(
        sum(safe_float(value) for value in values),
        2,
    )


def mean_values(
    values: list[Any],
    default: float = 0.0,
) -> float:
    numeric = [
        safe_float(value)
        for value in values
        if value is not None
    ]

    if not numeric:
        return default

    return round(float(np.mean(numeric)), 2)


def max_values(
    values: list[Any],
    default: float = 0.0,
) -> float:
    numeric = [
        safe_float(value)
        for value in values
        if value is not None
    ]

    if not numeric:
        return default

    return round(float(np.max(numeric)), 2)


def min_values(
    values: list[Any],
    default: float = 0.0,
) -> float:
    numeric = [
        safe_float(value)
        for value in values
        if value is not None
    ]

    if not numeric:
        return default

    return round(float(np.min(numeric)), 2)


def clean_barangay_name(name: str) -> str:
    cleaned = " ".join(str(name).strip().split())

    mapping = {
        "New Zaniga": "New Zañiga",
        "Old Zaniga": "Old Zañiga",
        "Pagasa": "Pag-Asa",
        "Pag-asa": "Pag-Asa",
        "Mabini J. Rizal": "Mabini-J. Rizal",
        "Mabini J Rizal": "Mabini-J. Rizal",
        "Plainiew": "Plainview",
        "Boni": "Plainview",
    }

    return mapping.get(cleaned, cleaned)


def get_weather_description(
    weather_code: int,
) -> dict[str, str]:
    if weather_code == 0:
        return {"condition": "Clear sky", "icon": "☀️"}
    if weather_code in [1, 2]:
        return {"condition": "Partly cloudy", "icon": "🌤️"}
    if weather_code == 3:
        return {"condition": "Overcast", "icon": "☁️"}
    if weather_code in [45, 48]:
        return {"condition": "Foggy", "icon": "🌫️"}
    if weather_code in [51, 53, 55, 56, 57]:
        return {"condition": "Drizzle", "icon": "🌦️"}
    if weather_code in [61, 63, 65, 66, 67]:
        return {"condition": "Rain", "icon": "🌧️"}
    if weather_code in [80, 81, 82]:
        return {"condition": "Rain showers", "icon": "🌦️"}
    if weather_code in [95, 96, 99]:
        return {"condition": "Thunderstorm", "icon": "⛈️"}

    return {"condition": "Unknown weather", "icon": "🌡️"}


def build_severity_input(
    profile: BarangayProfile,
    window: dict[str, Any],
    event_time: datetime,
    storm_signal: int,
    wind_direction_deg: float,
) -> pd.DataFrame:
    rainfall_24h = safe_float(window["rainfall_24h_mm"])
    rainfall_3d = safe_float(window["rainfall_3d_mm"])
    rainfall_7d = safe_float(window["rainfall_7d_mm"])

    elevation = safe_float(profile.elevation_m)
    distance = safe_float(profile.distance_to_waterway_m)

    return pd.DataFrame(
        [
            {
                "MONTH": event_time.month,
                "HOUR": event_time.hour,
                "IS_WEEKEND": int(event_time.weekday() >= 5),
                "WET_SEASON": int(5 <= event_time.month <= 11),
                "BARANGAY": clean_barangay_name(profile.barangay),
                "NEAREST_WATERWAY": profile.nearest_waterway,
                "ELEVATION_M": elevation,
                "DISTANCE_TO_WATERWAY_M": distance,
                "RAINFALL_24H_MM": rainfall_24h,
                "RAINFALL_3D_MM": rainfall_3d,
                "RAINFALL_7D_MM": rainfall_7d,
                "RAIN_PREV_2D_MM": max(
                    0.0,
                    rainfall_3d - rainfall_24h,
                ),
                "RAIN_PREV_4D_MM": max(
                    0.0,
                    rainfall_7d - rainfall_3d,
                ),
                "RAIN_PER_ELEV": rainfall_3d / (elevation + 1.0),
                "RAIN_DISTANCE_PRESSURE": (
                    rainfall_3d / (distance + 50.0)
                ),
                "TEMPERATURE_C": window["temperature_mean_c"],
                "TEMP_MAX_C": window["temperature_max_c"],
                "TEMP_MIN_C": window["temperature_min_c"],
                "WIND_SPEED_KPH": window["wind_speed_max_kph"],
                "WIND_DIRECTION_DEG": wind_direction_deg,
                "STORM_SIGNAL": storm_signal,
            }
        ]
    )


# ============================================================
# LIVE WEATHER
# ============================================================

async def fetch_live_weather(
    force_refresh: bool = False,
) -> dict[str, Any]:
    global _weather_cache
    global _weather_cache_fetched_at

    manila_tz = ZoneInfo(MANILA_TIMEZONE)
    now = datetime.now(manila_tz)

    def cache_is_fresh() -> bool:
        if (
            _weather_cache is None
            or _weather_cache_fetched_at is None
        ):
            return False

        # Reject a cache created by the older API version because it does
        # not contain the 24/48/72-hour prediction windows required here.
        prediction_windows = _weather_cache.get("prediction_windows")
        forecast_windows = _weather_cache.get("forecast_windows")

        if (
            not isinstance(prediction_windows, list)
            or len(prediction_windows) < 3
            or not isinstance(forecast_windows, dict)
            or not all(
                str(hours) in forecast_windows
                for hours in (24, 48, 72)
            )
        ):
            return False

        return (
            now - _weather_cache_fetched_at
            < timedelta(minutes=WEATHER_CACHE_TTL_MINUTES)
        )

    def cached_response(
        status: str,
        fallback_reason: str | None = None,
    ) -> dict[str, Any]:
        if _weather_cache is None:
            raise RuntimeError("Weather cache is empty.")

        result = deepcopy(_weather_cache)
        cache_age_seconds = 0

        if _weather_cache_fetched_at is not None:
            cache_age_seconds = max(
                int(
                    (
                        datetime.now(manila_tz)
                        - _weather_cache_fetched_at
                    ).total_seconds()
                ),
                0,
            )

        result["cache_status"] = status
        result["cache_age_seconds"] = cache_age_seconds
        result["cache_ttl_minutes"] = WEATHER_CACHE_TTL_MINUTES

        if fallback_reason:
            result["weather_warning"] = fallback_reason
        else:
            result.pop("weather_warning", None)

        return result

    if not force_refresh and cache_is_fresh():
        return cached_response("fresh-cache")

    async with _weather_cache_lock:
        now = datetime.now(manila_tz)

        if not force_refresh and cache_is_fresh():
            return cached_response("fresh-cache")

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
                    "wind_direction_10m",
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
            async with httpx.AsyncClient(timeout=30.0) as client:
                response: httpx.Response | None = None

                for attempt in range(OPEN_METEO_MAX_ATTEMPTS):
                    response = await client.get(url, params=params)

                    if response.status_code != 429:
                        break

                    if attempt < OPEN_METEO_MAX_ATTEMPTS - 1:
                        retry_after = response.headers.get("Retry-After")
                        delay_seconds = (
                            safe_float(retry_after, 0.0)
                            if retry_after
                            else float(2 ** attempt)
                        )
                        await asyncio.sleep(max(delay_seconds, 1.0))

                if response is None:
                    raise httpx.RequestError(
                        "Open-Meteo returned no response."
                    )

                response.raise_for_status()

        except httpx.TimeoutException as exc:
            if _weather_cache is not None:
                return cached_response(
                    "stale-fallback",
                    "Open-Meteo timed out. Using the last successful weather data.",
                )

            raise HTTPException(
                status_code=504,
                detail="Weather API request timed out.",
            ) from exc

        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code

            if _weather_cache is not None:
                return cached_response(
                    "stale-fallback",
                    (
                        f"Open-Meteo returned HTTP {status_code}. "
                        "Using the last successful weather data."
                    ),
                )

            raise HTTPException(
                status_code=503 if status_code == 429 else 502,
                detail=(
                    f"Unable to retrieve weather data: "
                    f"Open-Meteo returned HTTP {status_code}."
                ),
            ) from exc

        except httpx.RequestError as exc:
            if _weather_cache is not None:
                return cached_response(
                    "stale-fallback",
                    "Open-Meteo is temporarily unreachable. Using cached weather data.",
                )

            raise HTTPException(
                status_code=502,
                detail=f"Unable to retrieve weather data: {exc}",
            ) from exc

        payload = response.json()
        current = payload.get("current", {})
        hourly = payload.get("hourly", {})

        times = hourly.get("time", [])
        precipitation = hourly.get("precipitation", [])
        hourly_temperature = hourly.get("temperature_2m", [])
        hourly_humidity = hourly.get("relative_humidity_2m", [])
        hourly_wind = hourly.get("wind_speed_10m", [])

        if not times or not precipitation:
            if _weather_cache is not None:
                return cached_response(
                    "stale-fallback",
                    "Open-Meteo returned incomplete data. Using cached weather data.",
                )

            raise HTTPException(
                status_code=502,
                detail="Weather API returned incomplete data.",
            )

        try:
            parsed_times = [
                datetime.fromisoformat(value)
                for value in times
            ]
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=502,
                detail="Weather API returned invalid time data.",
            ) from exc

        manila_now = datetime.now(manila_tz)
        naive_now = manila_now.replace(
            tzinfo=None,
            minute=0,
            second=0,
            microsecond=0,
        )

        current_index = min(
            range(len(parsed_times)),
            key=lambda index: abs(parsed_times[index] - naive_now),
        )

        weather_code = int(
            safe_float(current.get("weather_code"), 0)
        )
        weather_description = get_weather_description(weather_code)

        wind_direction_deg = round(
            safe_float(current.get("wind_direction_10m"), 0.0),
            2,
        )

        prediction_windows: list[dict[str, Any]] = []

        for window_number, offset in enumerate([0, 24, 48], start=1):
            start_index = current_index + offset
            end_index = min(start_index + 24, len(parsed_times))

            if start_index >= len(parsed_times):
                break

            target_precip = precipitation[start_index:end_index]
            target_temp = hourly_temperature[start_index:end_index]
            target_humidity = hourly_humidity[start_index:end_index]
            target_wind = hourly_wind[start_index:end_index]

            rainfall_24h = sum_values(target_precip)
            rainfall_3d = sum_values(
                precipitation[max(0, start_index - 48):end_index]
            )
            rainfall_7d = sum_values(
                precipitation[max(0, start_index - 144):end_index]
            )

            start_time_naive = parsed_times[start_index]
            start_time = start_time_naive.replace(tzinfo=manila_tz)
            end_time = start_time + timedelta(hours=24)

            prediction_windows.append(
                {
                    "window_number": window_number,
                    "hours_from_now_start": offset,
                    "hours_from_now_end": offset + 24,
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "start_display": start_time.strftime(
                        "%b %d, %Y %I:%M %p"
                    ),
                    "end_display": end_time.strftime(
                        "%b %d, %Y %I:%M %p"
                    ),
                    "rainfall_24h_mm": rainfall_24h,
                    "rainfall_3d_mm": rainfall_3d,
                    "rainfall_7d_mm": rainfall_7d,
                    "max_hourly_rain_mm": max_values(target_precip),
                    "temperature_mean_c": mean_values(
                        target_temp,
                        safe_float(current.get("temperature_2m")),
                    ),
                    "temperature_max_c": max_values(
                        target_temp,
                        safe_float(current.get("temperature_2m")),
                    ),
                    "temperature_min_c": min_values(
                        target_temp,
                        safe_float(current.get("temperature_2m")),
                    ),
                    "humidity_mean_pct": mean_values(
                        target_humidity,
                        safe_float(current.get("relative_humidity_2m")),
                    ),
                    "wind_speed_mean_kph": mean_values(
                        target_wind,
                        safe_float(current.get("wind_speed_10m")),
                    ),
                    "wind_speed_max_kph": max_values(
                        target_wind,
                        safe_float(current.get("wind_speed_10m")),
                    ),
                }
            )

        if len(prediction_windows) < 3:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Open-Meteo did not return enough hourly forecast data "
                    "for all 72 hours."
                ),
            )

        forecast_windows: dict[str, dict[str, Any]] = {}

        for hours in [24, 48, 72]:
            count = hours // 24
            selected = prediction_windows[:count]
            forecast_windows[str(hours)] = {
                "hours": hours,
                "start": selected[0]["start"],
                "end": selected[-1]["end"],
                "start_display": selected[0]["start_display"],
                "end_display": selected[-1]["end_display"],
                "rainfall_mm": round(
                    sum(w["rainfall_24h_mm"] for w in selected),
                    2,
                ),
            }

        previous_values = precipitation[: current_index + 1]
        observed_24h = sum_values(previous_values[-24:])
        observed_3d = sum_values(previous_values[-72:])
        observed_7d = sum_values(previous_values[-168:])

        window24 = prediction_windows[0]

        weather = {
            "source": "Open-Meteo",
            "current_date": manila_now.strftime("%B %d, %Y"),
            "current_time": manila_now.strftime("%I:%M:%S %p"),
            "generated_at": manila_now.isoformat(),
            "forecast_horizon": "Selectable 24 / 48 / 72 hours",
            "condition": weather_description["condition"],
            "weather_icon": weather_description["icon"],
            "weather_code": weather_code,
            "temperature_c": round(
                safe_float(current.get("temperature_2m")),
                2,
            ),
            "humidity_pct": round(
                safe_float(current.get("relative_humidity_2m")),
                2,
            ),
            "wind_speed_kph": round(
                safe_float(current.get("wind_speed_10m")),
                2,
            ),
            "wind_direction_deg": wind_direction_deg,
            "current_precipitation_mm": round(
                safe_float(current.get("precipitation")),
                2,
            ),
            "observed_rainfall": {
                "past_24h_mm": observed_24h,
                "past_3d_mm": observed_3d,
                "past_7d_mm": observed_7d,
            },
            "forecast_rainfall": {
                "next_24h_mm": forecast_windows["24"]["rainfall_mm"],
                "next_48h_mm": forecast_windows["48"]["rainfall_mm"],
                "next_72h_mm": forecast_windows["72"]["rainfall_mm"],
            },
            "forecast_windows": forecast_windows,
            "prediction_windows": prediction_windows,
            "v2_features": {
                "temperature_mean_c": window24["temperature_mean_c"],
                "temperature_max_c": window24["temperature_max_c"],
                "temperature_min_c": window24["temperature_min_c"],
                "humidity_mean_pct": window24["humidity_mean_pct"],
                "daily_rain_mm": window24["rainfall_24h_mm"],
                "max_hourly_rain_mm": window24["max_hourly_rain_mm"],
                "rainfall_24h_mm": window24["rainfall_24h_mm"],
                "rainfall_3d_mm": window24["rainfall_3d_mm"],
                "rainfall_7d_mm": window24["rainfall_7d_mm"],
                "wind_speed_mean_kph": window24["wind_speed_mean_kph"],
                "wind_speed_max_kph": window24["wind_speed_max_kph"],
            },
            "storm_signal": 0,
            "storm_signal_source": (
                "No automatic official PAGASA signal connected"
            ),
            "tide_level_m": 0.0,
            "tide_source": "No automatic tide source connected",
        }

        _weather_cache = deepcopy(weather)
        _weather_cache_fetched_at = manila_now
        persist_weather_cache(
            _weather_cache,
            _weather_cache_fetched_at,
        )

        return cached_response("live-refresh")


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home() -> dict[str, Any]:
    return {
        "success": True,
        "message": "M.A.P.S. ML API is running.",
        "version": "2.2.0",
        "status": "running",
        "supported_forecast_hours": [24, 48, 72],
        "endpoints": {
            "health": "/health",
            "live_weather": "/weather/live",
            "citywide_prediction": "/predict/citywide",
            "documentation": "/docs",
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    severity_model_loaded = flood_code_classifier is not None

    return {
        "status": "healthy" if severity_model_loaded else "degraded",
        "api_version": "2.2.0",
        "prediction_type": "severity_only",
        "supported_forecast_hours": [24, 48, 72],
        "flood_severity_model_loaded": severity_model_loaded,
        "flood_severity_model": "maps_flood_code_classifier.joblib",
        "classes": ["A", "B", "C", "D"],
        "metadata_loaded": True,
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

    forecast_hours = int(request.forecast_hours)

    if forecast_hours not in ALLOWED_FORECAST_HOURS:
        raise HTTPException(
            status_code=422,
            detail="forecast_hours must be 24, 48, or 72.",
        )

    weather = await fetch_live_weather()
    prediction_windows = weather.get("prediction_windows", [])
    windows_needed = forecast_hours // 24
    selected_windows = deepcopy(prediction_windows[:windows_needed])

    if len(selected_windows) != windows_needed:
        raise HTTPException(
            status_code=502,
            detail=(
                "Weather data does not contain enough 24-hour prediction "
                "windows for the selected forecast period."
            ),
        )

    simulation = request.simulation

    if simulation is not None:
        if forecast_hours != 24:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Rainfall simulation supports a 24-hour severity "
                    "scenario only."
                ),
            )

        rainfall_24h = safe_float(simulation.rainfall_24h_mm)
        rainfall_3d = safe_float(simulation.rainfall_3d_mm)
        rainfall_7d = safe_float(simulation.rainfall_7d_mm)

        if rainfall_24h < 0 or rainfall_3d < 0 or rainfall_7d < 0:
            raise HTTPException(
                status_code=422,
                detail="Rainfall values cannot be negative.",
            )

        if rainfall_3d < rainfall_24h:
            raise HTTPException(
                status_code=422,
                detail=(
                    "3-day rainfall must be greater than or equal to "
                    "24-hour rainfall."
                ),
            )

        if rainfall_7d < rainfall_3d:
            raise HTTPException(
                status_code=422,
                detail=(
                    "7-day rainfall must be greater than or equal to "
                    "3-day rainfall."
                ),
            )

        # Keep temperature and wind from the live weather window, but
        # replace rainfall with the operator's hypothetical accumulation.
        selected_windows[0]["rainfall_24h_mm"] = rainfall_24h
        selected_windows[0]["rainfall_3d_mm"] = rainfall_3d
        selected_windows[0]["rainfall_7d_mm"] = rainfall_7d

    storm_signal = int(safe_float(weather.get("storm_signal", 0)))
    wind_direction_deg = safe_float(
        weather.get("wind_direction_deg", 0.0)
    )

    results: list[dict[str, Any]] = []

    for profile in request.barangays:
        window_results: list[dict[str, Any]] = []

        try:
            for window in selected_windows:
                event_time = datetime.fromisoformat(
                    str(window["start"])
                )

                severity_input = build_severity_input(
                    profile,
                    window,
                    event_time,
                    storm_signal,
                    wind_direction_deg,
                )

                severity_code = str(
                    flood_code_classifier.predict(
                        severity_input
                    )[0]
                )

                severity_probability_values = (
                    flood_code_classifier.predict_proba(
                        severity_input
                    )[0]
                )

                severity_classes = [
                    str(value)
                    for value in flood_code_classifier.classes_
                ]

                severity_probabilities = {
                    code: round(float(probability), 6)
                    for code, probability in zip(
                        severity_classes,
                        severity_probability_values,
                    )
                }

                severity_confidence = (
                    severity_probabilities.get(
                        severity_code,
                        0.0,
                    )
                )

                window_results.append(
                    {
                        "window_number": window["window_number"],
                        "start": window["start"],
                        "end": window["end"],
                        "start_display": window["start_display"],
                        "end_display": window["end_display"],
                        "rainfall_24h_mm": window["rainfall_24h_mm"],
                        "flood_code": severity_code,
                        "flood_severity": FLOOD_SEVERITY_LABELS.get(
                            severity_code,
                            "Unknown flood severity",
                        ),
                        "flood_severity_confidence": (
                            severity_confidence
                        ),
                        "flood_severity_probabilities": (
                            severity_probabilities
                        ),
                    }
                )

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Severity prediction failed for barangay "
                    f"'{profile.barangay}': {exc}"
                ),
            ) from exc

        # For 48/72-hour forecasts, report the highest severity
        # predicted across the consecutive 24-hour model windows.
        # If severity is tied, prefer the window with higher confidence.
        selected_result = max(
            window_results,
            key=lambda item: (
                SEVERITY_RANK.get(item["flood_code"], 0),
                item["flood_severity_confidence"],
            ),
        )

        severity_code = selected_result["flood_code"]
        severity_probabilities = (
            selected_result["flood_severity_probabilities"]
        )
        severity_confidence = float(
            selected_result["flood_severity_confidence"]
        )

        # Compatibility values for the existing Laravel storage schema.
        # These are derived only from A-D severity; no occurrence model is used.
        risk_level = {
            "A": "Low",
            "B": "Medium",
            "C": "High",
            "D": "High",
        }.get(severity_code, "Low")

        risk_score = RISK_SCORE[risk_level]

        storage_probabilities = {
            "Low": round(
                float(severity_probabilities.get("A", 0.0)),
                6,
            ),
            "Medium": round(
                float(severity_probabilities.get("B", 0.0)),
                6,
            ),
            "High": round(
                float(severity_probabilities.get("C", 0.0))
                + float(severity_probabilities.get("D", 0.0)),
                6,
            ),
        }

        result = {
            "barangay_id": profile.barangay_id,
            "barangay": profile.barangay,
            "forecast_hours": forecast_hours,
            "prediction_type": "severity_only",
            "flood_code": severity_code,
            "flood_severity": selected_result["flood_severity"],
            "flood_severity_confidence": severity_confidence,
            "flood_severity_probabilities": severity_probabilities,
            "severity_window_start": selected_result["start"],
            "severity_window_end": selected_result["end"],
            "confidence": severity_confidence,

            # Kept only so the existing database storage layer does not break.
            # They are derived from severity, not another ML model.
            "risk_level": risk_level,
            "risk_score": risk_score,
            "probabilities": storage_probabilities,

            "forecast_window_results": window_results,
            "cluster_number": "Not used for live prediction",
            "profile": {
                "barangay_clean": clean_barangay_name(
                    profile.barangay
                ),
                "nearest_waterway": profile.nearest_waterway,
                "waterway_type": profile.waterway_type,
                "elevation_m": profile.elevation_m,
                "distance_to_waterway_m": (
                    profile.distance_to_waterway_m
                ),
                "drainage_index": profile.drainage_index,
                "historical_flood_count_5y": (
                    profile.historical_flood_count_5y
                ),
                "previous_floods_30d": profile.previous_floods_30d,
                "days_since_previous_flood": (
                    profile.days_since_previous_flood
                ),
            },
        }

        results.append(result)

    results.sort(
        key=lambda item: (
            SEVERITY_RANK.get(item["flood_code"], 0),
            item["flood_severity_confidence"],
        ),
        reverse=True,
    )

    for rank, result in enumerate(results, start=1):
        result["rank"] = rank

    severity_distribution = {
        code: sum(
            1
            for result in results
            if result["flood_code"] == code
        )
        for code in ["A", "B", "C", "D"]
    }

    if simulation is not None:
        selected_forecast = deepcopy(
            weather["forecast_windows"]["24"]
        )
        selected_forecast["rainfall_mm"] = round(
            safe_float(simulation.rainfall_24h_mm),
            2,
        )
        selected_forecast["simulation"] = True
    else:
        selected_forecast = weather["forecast_windows"][
            str(forecast_hours)
        ]

    simulation_context = None
    if simulation is not None:
        simulation_context = {
            "enabled": True,
            "date": datetime.now(
                ZoneInfo(MANILA_TIMEZONE)
            ).strftime("%Y-%m-%d"),
            "rainfall_24h_mm": round(
                safe_float(simulation.rainfall_24h_mm),
                2,
            ),
            "rainfall_3d_mm": round(
                safe_float(simulation.rainfall_3d_mm),
                2,
            ),
            "rainfall_7d_mm": round(
                safe_float(simulation.rainfall_7d_mm),
                2,
            ),
            "automatic_features": {
                "temperature_c": weather.get("temperature_c"),
                "wind_speed_kph": weather.get("wind_speed_kph"),
                "wind_direction_deg": weather.get(
                    "wind_direction_deg"
                ),
                "storm_signal": weather.get("storm_signal", 0),
            },
        }

    return {
        "success": True,
        "generated_at": datetime.now(
            ZoneInfo(MANILA_TIMEZONE)
        ).isoformat(),
        "forecast_hours": forecast_hours,
        "prediction_horizon": (
            "Rainfall severity simulation for today"
            if simulation is not None
            else f"Next {forecast_hours} hours"
        ),
        "prediction_type": "severity_only",
        "simulation": simulation_context,
        "selected_forecast": selected_forecast,
        "models": {
            "flood_severity": "maps_flood_code_classifier.joblib",
        },
        "weather": weather,
        "summary": {
            "total_barangays": len(results),
            "forecast_hours": forecast_hours,
            "severity_distribution": severity_distribution,
            "highest_severity_code": (
                results[0]["flood_code"]
                if results
                else None
            ),
            "highest_severity_confidence": (
                results[0]["flood_severity_confidence"]
                if results
                else None
            ),
        },
        "predictions": results,
    }

