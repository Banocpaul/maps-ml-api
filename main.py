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

# Weather data is shared by the dashboard and citywide prediction.
# Reusing one successful response prevents unnecessary Open-Meteo calls.
WEATHER_CACHE_TTL_MINUTES = 30
WEATHER_CACHE_FILE = APP_DIR / "weather_cache.json"
OPEN_METEO_MAX_ATTEMPTS = 3

_weather_cache: dict[str, Any] | None = None
_weather_cache_fetched_at: datetime | None = None
_weather_cache_lock = asyncio.Lock()


def load_persisted_weather_cache() -> tuple[
    dict[str, Any] | None,
    datetime | None,
]:
    """Load the last successful weather response after a process restart."""
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
    """Persist weather atomically for Render process restarts."""
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
        # In-memory caching still works if the filesystem is unavailable.
        temporary_file.unlink(missing_ok=True)


_weather_cache, _weather_cache_fetched_at = (
    load_persisted_weather_cache()
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="M.A.P.S. ML API",
    version="2.0.0",
    description=(
        "Live-weather flood occurrence, risk, depth, and duration "
        "prediction service for Mandaluyong City."
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


# Final V2 models built from the Open-Meteo + geographic enrichment workflow.
occurrence_v2_model = load_model(
    "maps_flood_occurrence_v2.joblib"
)

occurrence_v2_config = load_model(
    "maps_flood_occurrence_v2_config.joblib"
)

duration_v2_model = load_model(
    "maps_flood_duration_v2.joblib"
)

# Severity and depth models trained from the CDRRMO flood-code records.
# These are separate from the legacy models above: the classifier returns
# flood codes A-D and the regressor returns a single depth value in feet.
severity_model_bundle = load_model(
    "maps_flood_severity_model.joblib"
)

depth_regression_bundle = load_model(
    "maps_flood_depth_regression_model.joblib"
)

OCCURRENCE_V2_THRESHOLD = float(
    occurrence_v2_config.get("threshold", 0.18)
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

    # Legacy fields are still accepted for the older risk/depth models.
    drainage_index: float
    impervious_surface_ratio: float
    population_density_per_km2: float
    historical_flood_count_5y: int

    # V2 fields. Defaults keep existing Laravel requests backward-compatible.
    waterway_type: str = "Unknown"
    previous_floods_30d: int = 0
    days_since_previous_flood: float = 999.0


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

FLOOD_SEVERITY_LABELS = {
    "A": "Level A - Minor Flooding (1 ft)",
    "B": "Level B - Moderate Flooding (2 ft)",
    "C": "Level C - Severe Flooding (3 ft)",
    "D": "Level D - Critical Flooding (4 ft)",
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


def clean_barangay_name(name: str) -> str:
    """
    Standardize common Mandaluyong barangay name variants so that live
    requests match the categories used when the V2 models were trained.
    """
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


def build_flood_code_input(
    bundle: dict[str, Any],
    profile: BarangayProfile,
    weather: dict[str, Any],
    event_time: datetime,
) -> pd.DataFrame:
    """Build the 553-column input expected by the A-D and feet models."""
    observed = weather["observed_rainfall"]
    forecast = weather["forecast_rainfall"]
    v2_weather = weather["v2_features"]

    # Start with training medians, then replace every value available from
    # Open-Meteo and the barangay profile.
    row = dict(bundle["numeric_medians"])
    row.update(
        {
            "BARANGAY": profile.barangay,
            "STREET": "Unknown",
            "CORNER": "Unknown",
            "CAUSED": weather["condition"],
            "YEAR": event_time.year,
            "MONTH": event_time.month,
            "DAY": event_time.day,
            "HOUR": event_time.hour,
            "DAY_OF_WEEK": event_time.weekday(),
            "WET_SEASON": int(5 <= event_time.month <= 11),
            "PEAK_HOUR": int(14 <= event_time.hour <= 20),
            "PRIOR_BARANGAY_COUNT": profile.historical_flood_count_5y,
            "DAYS_SINCE_BARANGAY": profile.days_since_previous_flood,
            "LATITUDE": MANDALUYONG_LATITUDE,
            "LONGITUDE": MANDALUYONG_LONGITUDE,
            "TEMPERATURE_C": weather["temperature_c"],
            "TEMPERATURE_MAX_C": v2_weather["temperature_max_c"],
            "HUMIDITY_PCT": weather["humidity_pct"],
            "RAINFALL_24H_MM": forecast["next_24h_mm"],
            "RAIN_24H_MM": observed["past_24h_mm"],
            "WIND_SPEED_KPH": weather["wind_speed_kph"],
            "RAINFALL_3D_MM": (
                observed["past_3d_mm"] + forecast["next_3d_mm"]
            ),
            "RAINFALL_7D_MM": (
                observed["past_7d_mm"] + forecast["next_7d_mm"]
            ),
        }
    )

    raw = pd.DataFrame([row])
    encoded = pd.get_dummies(
        raw,
        columns=bundle["categorical_features"],
        dtype=int,
    )

    return encoded.reindex(
        columns=bundle["feature_columns"],
        fill_value=0,
    )


# ============================================================
# LIVE WEATHER
# ============================================================

async def fetch_live_weather(
    force_refresh: bool = False,
) -> dict[str, Any]:
    """
    Retrieve Mandaluyong weather from Open-Meteo.

    Protection added for production:
    - Fresh weather is cached for 15 minutes.
    - Concurrent requests share the same refresh operation.
    - If Open-Meteo is temporarily unavailable or returns HTTP 429,
      the last successful weather response is returned as a stale fallback.
    - The original weather fields used by Laravel and the prediction model
      are preserved.
    """
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

        return (
            now - _weather_cache_fetched_at
            < timedelta(
                minutes=WEATHER_CACHE_TTL_MINUTES
            )
        )

    def cached_response(
        status: str,
        fallback_reason: str | None = None,
    ) -> dict[str, Any]:
        if _weather_cache is None:
            raise RuntimeError(
                "Weather cache is empty."
            )

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

        # These are additive metadata fields. Existing Laravel/model fields
        # remain unchanged.
        result["cache_status"] = status
        result["cache_age_seconds"] = cache_age_seconds
        result["cache_ttl_minutes"] = (
            WEATHER_CACHE_TTL_MINUTES
        )

        if fallback_reason:
            result["weather_warning"] = fallback_reason
        else:
            result.pop(
                "weather_warning",
                None,
            )

        return result

    # Fast path: do not contact Open-Meteo while cached data is fresh.
    if not force_refresh and cache_is_fresh():
        return cached_response("fresh-cache")

    # Prevent a burst of simultaneous Laravel requests from all refreshing
    # Open-Meteo at the same time.
    async with _weather_cache_lock:
        now = datetime.now(manila_tz)

        # Another request may already have refreshed the cache while this
        # request was waiting for the lock.
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
                response: httpx.Response | None = None

                for attempt in range(OPEN_METEO_MAX_ATTEMPTS):
                    response = await client.get(
                        url,
                        params=params,
                    )

                    if response.status_code != 429:
                        break

                    if attempt < OPEN_METEO_MAX_ATTEMPTS - 1:
                        retry_after = response.headers.get(
                            "Retry-After"
                        )
                        delay_seconds = (
                            safe_float(retry_after, 0.0)
                            if retry_after
                            else float(2 ** attempt)
                        )
                        await asyncio.sleep(
                            max(delay_seconds, 1.0)
                        )

                if response is None:
                    raise httpx.RequestError(
                        "Open-Meteo returned no response."
                    )

                response.raise_for_status()

        except httpx.TimeoutException as exc:
            if _weather_cache is not None:
                return cached_response(
                    "stale-fallback",
                    (
                        "Open-Meteo timed out. "
                        "Using the last successful weather data."
                    ),
                )

            raise HTTPException(
                status_code=504,
                detail="Weather API request timed out.",
            ) from exc

        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code

            if _weather_cache is not None:
                if status_code == 429:
                    reason = (
                        "Open-Meteo rate limit reached (HTTP 429). "
                        "Using the last successful weather data."
                    )
                else:
                    reason = (
                        "Open-Meteo returned HTTP "
                        f"{status_code}. "
                        "Using the last successful weather data."
                    )

                return cached_response(
                    "stale-fallback",
                    reason,
                )

            if status_code == 429:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Open-Meteo rate limit reached (HTTP 429) "
                        "and no cached weather is available yet. "
                        "Please retry after a short interval."
                    ),
                ) from exc

            raise HTTPException(
                status_code=502,
                detail=(
                    "Unable to retrieve weather data: "
                    f"Open-Meteo returned HTTP {status_code}."
                ),
            ) from exc

        except httpx.RequestError as exc:
            if _weather_cache is not None:
                return cached_response(
                    "stale-fallback",
                    (
                        "Open-Meteo is temporarily unreachable. "
                        "Using the last successful weather data."
                    ),
                )

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
        hourly_temperature = hourly.get(
            "temperature_2m",
            [],
        )
        hourly_humidity = hourly.get(
            "relative_humidity_2m",
            [],
        )
        hourly_wind = hourly.get(
            "wind_speed_10m",
            [],
        )

        if not times or not precipitation:
            if _weather_cache is not None:
                return cached_response(
                    "stale-fallback",
                    (
                        "Open-Meteo returned incomplete weather data. "
                        "Using the last successful weather data."
                    ),
                )

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

        try:
            parsed_times = [
                datetime.fromisoformat(value)
                for value in times
            ]
        except (TypeError, ValueError) as exc:
            if _weather_cache is not None:
                return cached_response(
                    "stale-fallback",
                    (
                        "Open-Meteo returned invalid time data. "
                        "Using the last successful weather data."
                    ),
                )

            raise HTTPException(
                status_code=502,
                detail=(
                    "Weather API returned invalid time data."
                ),
            ) from exc

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

        # V2 live feature window. We include the current hour plus the next
        # 23 hours so that the model receives a full 24-hour forecast window.
        window_start = current_index
        window_end = min(
            current_index + 24,
            len(parsed_times),
        )

        next_24h_precip = precipitation[
            window_start:window_end
        ]
        next_24h_temperature = hourly_temperature[
            window_start:window_end
        ]
        next_24h_humidity = hourly_humidity[
            window_start:window_end
        ]
        next_24h_wind = hourly_wind[
            window_start:window_end
        ]

        v2_daily_rain = sum_values(
            next_24h_precip
        )
        v2_max_hourly_rain = max_values(
            next_24h_precip
        )
        v2_temperature_mean = mean_values(
            next_24h_temperature,
            safe_float(current.get("temperature_2m")),
        )
        v2_temperature_max = max_values(
            next_24h_temperature,
            safe_float(current.get("temperature_2m")),
        )
        v2_humidity_mean = mean_values(
            next_24h_humidity,
            safe_float(current.get("relative_humidity_2m")),
        )
        v2_wind_mean = mean_values(
            next_24h_wind,
            safe_float(current.get("wind_speed_10m")),
        )
        v2_wind_max = max_values(
            next_24h_wind,
            safe_float(current.get("wind_speed_10m")),
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

        weather = {
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

            # Features consumed directly by the final V2 sklearn pipelines.
            "v2_features": {
                "temperature_mean_c": v2_temperature_mean,
                "temperature_max_c": v2_temperature_max,
                "humidity_mean_pct": v2_humidity_mean,
                "daily_rain_mm": v2_daily_rain,
                "max_hourly_rain_mm": v2_max_hourly_rain,
                "rainfall_24h_mm": v2_daily_rain,
                "rainfall_3d_mm": observed_3d,
                "rainfall_7d_mm": observed_7d,
                "wind_speed_mean_kph": v2_wind_mean,
                "wind_speed_max_kph": v2_wind_max,
            },

            # These values remain exactly as in the current project until
            # official automatic PAGASA and tide integrations are connected.
            "storm_signal": 0,
            "storm_signal_source":
                "No automatic official PAGASA signal connected",

            "tide_level_m": 0.0,
            "tide_source":
                "No automatic tide source connected",
        }

        # Save ONLY a successful, complete Open-Meteo response.
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
        "version": "2.0.0",
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
    """Health endpoint used by Laravel and Render."""
    occurrence_v2_loaded = occurrence_v2_model is not None
    duration_v2_loaded = duration_v2_model is not None
    severity_model_loaded = severity_model_bundle is not None
    depth_feet_model_loaded = depth_regression_bundle is not None

    healthy = (
        occurrence_v2_loaded
        and duration_v2_loaded
        and severity_model_loaded
        and depth_feet_model_loaded
    )

    return {
        "status": "healthy" if healthy else "degraded",
        "api_version": "2.0.0",
        "occurrence_v2_loaded": occurrence_v2_loaded,
        "duration_v2_loaded": duration_v2_loaded,
        "occurrence_threshold": OCCURRENCE_V2_THRESHOLD,
        "legacy_risk_model_loaded": False,
        "legacy_depth_model_loaded": False,
        "flood_severity_model_loaded": severity_model_loaded,
        "flood_depth_feet_model_loaded": depth_feet_model_loaded,
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

    weather = await fetch_live_weather()

    manila_now = datetime.now(
        ZoneInfo(MANILA_TIMEZONE)
    )

    month = manila_now.month
    day_of_week = manila_now.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    wet_season = 1 if 5 <= month <= 11 else 0

    storm_signal = int(
        weather["storm_signal"]
    )
    tide_level_m = float(
        weather["tide_level_m"]
    )

    observed = weather["observed_rainfall"]
    forecast = weather["forecast_rainfall"]
    v2_weather = weather["v2_features"]

    results: list[dict[str, Any]] = []

    for profile in request.barangays:
        barangay_clean = clean_barangay_name(
            profile.barangay
        )

        # --------------------------------------------------------
        # FINAL V2 INPUT
        # The occurrence and duration V2 pipelines were intentionally
        # trained with the same 22 raw input fields.
        # --------------------------------------------------------
        v2_input = pd.DataFrame(
            [
                {
                    "temperature_mean_c":
                        v2_weather["temperature_mean_c"],
                    "temperature_max_c":
                        v2_weather["temperature_max_c"],
                    "humidity_mean_pct":
                        v2_weather["humidity_mean_pct"],
                    "daily_rain_mm":
                        v2_weather["daily_rain_mm"],
                    "max_hourly_rain_mm":
                        v2_weather["max_hourly_rain_mm"],
                    "rainfall_24h_mm":
                        v2_weather["rainfall_24h_mm"],
                    "rainfall_3d_mm":
                        v2_weather["rainfall_3d_mm"],
                    "rainfall_7d_mm":
                        v2_weather["rainfall_7d_mm"],
                    "wind_speed_mean_kph":
                        v2_weather["wind_speed_mean_kph"],
                    "wind_speed_max_kph":
                        v2_weather["wind_speed_max_kph"],
                    "month": month,
                    "day_of_week": day_of_week,
                    "is_weekend": is_weekend,
                    "wet_season": wet_season,
                    "historical_flood_count":
                        profile.historical_flood_count_5y,
                    "previous_floods_30d":
                        profile.previous_floods_30d,
                    "days_since_previous_flood":
                        profile.days_since_previous_flood,
                    "elevation_m":
                        profile.elevation_m,
                    "distance_to_waterway_m":
                        profile.distance_to_waterway_m,
                    "BARANGAY_CLEAN":
                        barangay_clean,
                    "nearest_waterway":
                        profile.nearest_waterway,
                    "waterway_type":
                        profile.waterway_type,
                }
            ]
        )

        flood_code_input = build_flood_code_input(
            severity_model_bundle,
            profile,
            weather,
            manila_now,
        )

        try:
            # ---------------- FINAL OCCURRENCE V2 ----------------
            occurrence_probability = float(
                occurrence_v2_model.predict_proba(
                    v2_input
                )[0, 1]
            )

            flood_predicted = (
                occurrence_probability
                >= OCCURRENCE_V2_THRESHOLD
            )

            occurrence_label = (
                "Flood"
                if flood_predicted
                else "No Flood"
            )

            # ---------------- PRIMARY V2 RISK --------------------
            # Keep the main UI consistent with Occurrence V2.
            if not flood_predicted:
                risk_level_v2 = "Low"
                risk_score_v2 = 1
            elif occurrence_probability >= 0.50:
                risk_level_v2 = "High"
                risk_score_v2 = 3
            else:
                risk_level_v2 = "Medium"
                risk_score_v2 = 2

            # ---------------- FINAL DURATION V2 ------------------
            predicted_duration_v2 = max(
                float(
                    duration_v2_model.predict(
                        v2_input
                    )[0]
                ),
                0.0,
            )

            # ---------------- FLOOD CODE + DEPTH IN FEET --------
            # The A-D code and feet estimate are supplemental outputs from
            # the latest CDRRMO-record model. Existing fields remain intact.
            severity_probabilities = (
                severity_model_bundle["model"].predict_proba(
                    flood_code_input
                )[0]
            )
            severity_code = str(
                severity_model_bundle["model"].classes_[
                    int(np.argmax(severity_probabilities))
                ]
            )
            severity_probability_map = {
                str(code): round(float(probability), 6)
                for code, probability in zip(
                    severity_model_bundle["model"].classes_,
                    severity_probabilities,
                )
            }
            predicted_depth_ft = max(
                float(
                    depth_regression_bundle["model"].predict(
                        build_flood_code_input(
                            depth_regression_bundle,
                            profile,
                            weather,
                            manila_now,
                        )
                    )[0]
                ),
                0.0,
            )

            # Compatibility values allow the existing Laravel views to keep
            # working without retaining two large legacy models in memory.
            predicted_depth_mm = predicted_depth_ft * 304.8
            legacy_predicted_duration = predicted_duration_v2
            probabilities = {
                "Low": round(1.0 - occurrence_probability, 6),
                "Medium": round(occurrence_probability, 6),
                "High": round(occurrence_probability, 6),
            }

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

                # Final V2 occurrence output.
                "flood_occurrence":
                    occurrence_label,

                "flood_predicted":
                    flood_predicted,

                "flood_probability":
                    round(
                        occurrence_probability,
                        6,
                    ),

                "occurrence_threshold":
                    round(
                        OCCURRENCE_V2_THRESHOLD,
                        4,
                    ),

                "occurrence_model":
                    "MAPS Flood Occurrence V2",

                # Primary risk fields now follow Occurrence V2 so that
                # the UI cannot show "No Flood" together with Medium/High risk.
                "risk_level":
                    risk_level_v2,

                "risk_score":
                    risk_score_v2,

                # Legacy classifier values are preserved for comparison and
                # backward-compatible analytics/debugging. They now mirror
                # the V2 occurrence result to avoid loading the old model.
                "legacy_risk_level":
                    risk_level_v2,

                "legacy_risk_score":
                    risk_score_v2,

                "legacy_confidence":
                    round(
                        occurrence_probability
                        if flood_predicted
                        else 1.0 - occurrence_probability,
                        6,
                    ),

                "legacy_probabilities":
                    probabilities,

                # Latest flood-code classifier and tuned depth regressor.
                "flood_code": severity_code,
                "flood_severity": FLOOD_SEVERITY_LABELS.get(
                    severity_code,
                    "Unknown flood severity",
                ),
                "flood_severity_confidence": round(
                    max(severity_probabilities),
                    6,
                ),
                "flood_severity_probabilities": severity_probability_map,
                "predicted_depth_ft": round(predicted_depth_ft, 2),

               # Primary confidence now follows Flood Occurrence V2.
# For Flood, confidence equals the flood probability.
# For No Flood, confidence equals 1 minus the flood probability.
"confidence":
    round(
        occurrence_probability
        if flood_predicted
        else 1.0 - occurrence_probability,
        6,
    ),

# Explicit binary probabilities from Flood Occurrence V2.
"occurrence_probabilities": {
    "Flood":
        round(
            occurrence_probability,
            6,
        ),
    "No Flood":
        round(
            1.0 - occurrence_probability,
            6,
        ),
},

# Retained temporarily because the existing Laravel interface
# may still expect High, Medium, and Low probability keys.
"probabilities":
    probabilities,

                # Depth remains from the legacy regression model until a
                # replacement depth model is validated.
                "predicted_depth_mm":
                    round(predicted_depth_mm, 2),

                # Final V2 duration replaces the old duration value.
                "predicted_duration_hours":
                    round(
                        predicted_duration_v2,
                        2,
                    ),

                "duration_model":
                    "MAPS Flood Duration V2",

                # Included temporarily for comparison/debugging.
                "legacy_predicted_duration_hours":
                    legacy_predicted_duration,

                "cluster_number":
                    "Not used for live prediction",

                "profile": {
                    "barangay_clean":
                        barangay_clean,
                    "nearest_waterway":
                        profile.nearest_waterway,
                    "waterway_type":
                        profile.waterway_type,
                    "elevation_m":
                        profile.elevation_m,
                    "distance_to_waterway_m":
                        profile.distance_to_waterway_m,
                    "drainage_index":
                        profile.drainage_index,
                    "historical_flood_count_5y":
                        profile.historical_flood_count_5y,
                    "previous_floods_30d":
                        profile.previous_floods_30d,
                    "days_since_previous_flood":
                        profile.days_since_previous_flood,
                },
            }
        )

    # V2 occurrence probability is now the primary citywide ranking signal.
    # Legacy risk/depth are secondary tie-breakers for compatibility.
    results.sort(
        key=lambda item: (
            int(item["flood_predicted"]),
            item["flood_probability"],
            item["risk_score"],
            item["predicted_depth_ft"],
            item["predicted_depth_ft"],
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

    occurrence_distribution = {
        "Flood": sum(
            1
            for result in results
            if result["flood_predicted"]
        ),
        "No Flood": sum(
            1
            for result in results
            if not result["flood_predicted"]
        ),
    }

    return {
        "success": True,

        "generated_at":
            manila_now.isoformat(),

        "prediction_horizon":
            "Next 24 hours",

        "models": {
            "occurrence":
                "MAPS Flood Occurrence V2",
            "duration":
                "MAPS Flood Duration V2",
            "legacy_risk":
                "Replaced by MAPS Flood Occurrence V2",
            "legacy_depth":
                "Replaced by maps_flood_depth_regression_model.joblib",
            "flood_severity":
                "maps_flood_severity_model.joblib",
            "flood_depth_feet":
                "maps_flood_depth_regression_model.joblib",
        },

        "weather":
            weather,

        "summary": {
            "total_barangays":
                len(results),

            "occurrence_threshold":
                OCCURRENCE_V2_THRESHOLD,

            "occurrence_distribution":
                occurrence_distribution,

            "predicted_flood_barangays":
                occurrence_distribution["Flood"],

            "highest_flood_probability":
                max(
                    item["flood_probability"]
                    for item in results
                ),

            # Legacy summary retained for Laravel compatibility.
            "risk_distribution":
                risk_distribution,

            "highest_predicted_depth_mm":
                max(
                    item["predicted_depth_mm"]
                    for item in results
                ),

            "highest_predicted_depth_ft": max(
                item["predicted_depth_ft"]
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
