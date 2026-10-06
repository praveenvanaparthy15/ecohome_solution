from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_core.tools import tool
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from models.energy import EnergyUsage, SolarGeneration


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "data" / "energy_data.db"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

engine = create_engine(
    DATABASE_URL,
    echo=False,
)

logger = logging.getLogger("ecohome.tools")

logging.basicConfig(
    filename=PROJECT_ROOT / "ecohome_agent.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float:
    """Convert a value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _log_tool(tool_name: str, details: Any) -> None:
    """Log tool execution for transparency."""
    logger.info("%s | %s", tool_name, details)


# ---------------------------------------------------------------------------
# Weather Tool
# ---------------------------------------------------------------------------

@tool
def get_weather_forecast(
    date: str,
) -> dict[str, Any]:
    """
    Get a mock weather forecast for a requested date.

    The project intentionally uses deterministic mock weather data so that
    the agent can be evaluated without requiring an external API key.

    Args:
        date: Date in YYYY-MM-DD format.

    Returns:
        Weather forecast including temperature, condition, cloud cover,
        and expected solar-generation factor.
    """

    try:
        requested_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        return {
            "success": False,
            "error": "Invalid date. Use YYYY-MM-DD format.",
        }

    # Deterministic mock weather based on day-of-year.
    day_number = requested_date.timetuple().tm_yday

    conditions = [
        ("sunny", 32, 10, 1.00),
        ("partly_cloudy", 29, 35, 0.80),
        ("cloudy", 26, 70, 0.55),
        ("sunny", 34, 15, 0.95),
        ("partly_cloudy", 30, 40, 0.75),
    ]

    condition, temperature, cloud_cover, solar_factor = (
        conditions[day_number % len(conditions)]
    )

    result = {
        "success": True,
        "date": date,
        "temperature_c": temperature,
        "condition": condition,
        "cloud_cover_percent": cloud_cover,
        "solar_generation_factor": solar_factor,
        "source": "EcoHome mock weather service",
    }

    _log_tool("get_weather_forecast", result)

    return result


# ---------------------------------------------------------------------------
# Electricity Pricing Tool
# ---------------------------------------------------------------------------

@tool
def get_electricity_prices(
    date: str,
) -> dict[str, Any]:
    """
    Get time-of-use electricity prices for a requested date.

    Prices are deterministic mock values for evaluation.

    Args:
        date: Date in YYYY-MM-DD format.

    Returns:
        Hourly electricity pricing periods and rates.
    """

    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {
            "success": False,
            "error": "Invalid date. Use YYYY-MM-DD format.",
        }

    prices = []

    for hour in range(24):

        if 0 <= hour < 6:
            period = "off_peak"
            price = 0.08

        elif 6 <= hour < 17:
            period = "standard"
            price = 0.14

        elif 17 <= hour < 21:
            period = "peak"
            price = 0.25

        else:
            period = "standard"
            price = 0.14

        prices.append(
            {
                "hour": hour,
                "period": period,
                "price_per_kwh": price,
            }
        )

    result = {
        "success": True,
        "date": date,
        "currency": "USD",
        "prices": prices,
        "off_peak_rate": 0.08,
        "standard_rate": 0.14,
        "peak_rate": 0.25,
        "source": "EcoHome mock electricity pricing service",
    }

    _log_tool("get_electricity_prices", result)

    return result


# ---------------------------------------------------------------------------
# Historical Energy Usage Tool
# ---------------------------------------------------------------------------

@tool
def get_energy_usage(
    device_type: str | None = None,
    days: int = 7,
) -> dict[str, Any]:
    """
    Retrieve historical energy consumption from the EcoHome database.

    Args:
        device_type: Optional device such as EV, HVAC, Dishwasher,
                     Washing Machine, Pool Pump, or Water Heater.
        days: Number of historical days to analyze.

    Returns:
        Usage totals, costs, averages, and device-level statistics.
    """

    try:
        days = max(1, min(int(days), 30))
    except (TypeError, ValueError):
        days = 7

    cutoff = datetime.now() - timedelta(days=days)

    try:
        with Session(engine) as session:

            stmt = select(EnergyUsage).where(
                EnergyUsage.timestamp >= cutoff
            )

            if device_type:
                stmt = stmt.where(
                    func.lower(EnergyUsage.device_type)
                    == device_type.lower()
                )

            records = session.scalars(stmt).all()

            if not records:
                return {
                    "success": True,
                    "device_type": device_type,
                    "days": days,
                    "record_count": 0,
                    "total_energy_kwh": 0,
                    "total_cost": 0,
                    "message": "No matching energy records found.",
                }

            total_energy = sum(
                _safe_float(record.energy_kwh)
                for record in records
            )

            total_cost = sum(
                _safe_float(record.cost)
                for record in records
            )

            by_device: dict[str, dict[str, float]] = {}

            for record in records:

                device = record.device_type

                if device not in by_device:
                    by_device[device] = {
                        "energy_kwh": 0.0,
                        "cost": 0.0,
                    }

                by_device[device]["energy_kwh"] += (
                    _safe_float(record.energy_kwh)
                )

                by_device[device]["cost"] += (
                    _safe_float(record.cost)
                )

        result = {
            "success": True,
            "device_type": device_type,
            "days": days,
            "record_count": len(records),
            "total_energy_kwh": round(total_energy, 2),
            "total_cost": round(total_cost, 2),
            "average_daily_energy_kwh": round(
                total_energy / days,
                2,
            ),
            "average_daily_cost": round(
                total_cost / days,
                2,
            ),
            "by_device": {
                device: {
                    "energy_kwh": round(values["energy_kwh"], 2),
                    "cost": round(values["cost"], 2),
                }
                for device, values in by_device.items()
            },
        }

        _log_tool("get_energy_usage", result)

        return result

    except Exception as exc:
        logger.exception("Energy usage query failed")

        return {
            "success": False,
            "error": f"Unable to query energy usage: {exc}",
        }


# ---------------------------------------------------------------------------
# Solar Generation Tool
# ---------------------------------------------------------------------------

@tool
def get_solar_generation(
    days: int = 7,
) -> dict[str, Any]:
    """
    Retrieve historical solar generation data.

    Args:
        days: Number of days to analyze.

    Returns:
        Total solar generation and average daily generation.
    """

    try:
        days = max(1, min(int(days), 30))
    except (TypeError, ValueError):
        days = 7

    cutoff = datetime.now() - timedelta(days=days)

    try:
        with Session(engine) as session:

            stmt = select(SolarGeneration).where(
                SolarGeneration.timestamp >= cutoff
            )

            records = session.scalars(stmt).all()

            total_generation = sum(
                _safe_float(record.generation_kwh)
                for record in records
            )

            weather_counts: dict[str, int] = {}

            for record in records:
                condition = record.weather_condition

                weather_counts[condition] = (
                    weather_counts.get(condition, 0) + 1
                )

        result = {
            "success": True,
            "days": days,
            "record_count": len(records),
            "total_generation_kwh": round(
                total_generation,
                2,
            ),
            "average_daily_generation_kwh": round(
                total_generation / days,
                2,
            ),
            "weather_distribution": weather_counts,
        }

        _log_tool("get_solar_generation", result)

        return result

    except Exception as exc:
        logger.exception("Solar generation query failed")

        return {
            "success": False,
            "error": f"Unable to query solar generation: {exc}",
        }


# ---------------------------------------------------------------------------
# Energy Tips RAG Tool
# ---------------------------------------------------------------------------

@tool
def search_energy_tips(
    query: str,
    k: int = 3,
) -> dict[str, Any]:
    """
    Search the EcoHome energy-saving knowledge base using ChromaDB.

    Args:
        query: Energy optimization question.
        k: Number of relevant results.

    Returns:
        Relevant knowledge-base documents and their sources.
    """

    try:
        if not os.getenv("COHERE_API_KEY"):
            return {
                "success": False,
                "error": "COHERE_API_KEY is not configured.",
            }

        k = max(1, min(int(k), 5))

        embeddings = CohereEmbeddings(
            model="embed-english-v3.0"
        )

        vectorstore = Chroma(
            collection_name="ecohome_energy_tips",
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings,
        )

        results = vectorstore.similarity_search(
            query,
            k=k,
        )

        formatted_results = []

        for index, document in enumerate(results, 1):

            source = Path(
                document.metadata.get(
                    "source",
                    "unknown",
                )
            ).name

            formatted_results.append(
                {
                    "rank": index,
                    "source": source,
                    "content": document.page_content,
                }
            )

        result = {
            "success": True,
            "query": query,
            "result_count": len(formatted_results),
            "results": formatted_results,
        }

        _log_tool(
            "search_energy_tips",
            {
                "query": query,
                "result_count": len(formatted_results),
            },
        )

        return result

    except Exception as exc:
        logger.exception("RAG search failed")

        return {
            "success": False,
            "error": f"Unable to search energy tips: {exc}",
        }


# ---------------------------------------------------------------------------
# Savings Calculation Tool
# ---------------------------------------------------------------------------

@tool
def calculate_savings(
    energy_kwh: float,
    current_rate: float,
    optimized_rate: float,
) -> dict[str, Any]:
    """
    Calculate cost and savings from shifting electricity consumption.

    Args:
        energy_kwh: Energy consumed by the device.
        current_rate: Current electricity price per kWh.
        optimized_rate: Optimized electricity price per kWh.

    Returns:
        Current cost, optimized cost, savings, and percentage reduction.
    """

    try:
        energy = max(0.0, float(energy_kwh))
        current = max(0.0, float(current_rate))
        optimized = max(0.0, float(optimized_rate))

        current_cost = energy * current
        optimized_cost = energy * optimized
        savings = max(0.0, current_cost - optimized_cost)

        percentage = (
            (savings / current_cost) * 100
            if current_cost > 0
            else 0.0
        )

        result = {
            "success": True,
            "energy_kwh": round(energy, 2),
            "current_rate": round(current, 3),
            "optimized_rate": round(optimized, 3),
            "current_cost": round(current_cost, 2),
            "optimized_cost": round(optimized_cost, 2),
            "estimated_savings": round(savings, 2),
            "savings_percentage": round(percentage, 2),
        }

        _log_tool("calculate_savings", result)

        return result

    except Exception as exc:
        logger.exception("Savings calculation failed")

        return {
            "success": False,
            "error": f"Unable to calculate savings: {exc}",
        }


# ---------------------------------------------------------------------------
# Tool collection
# ---------------------------------------------------------------------------

ECOHOME_TOOLS = [
    get_weather_forecast,
    get_electricity_prices,
    get_energy_usage,
    get_solar_generation,
    search_energy_tips,
    calculate_savings,
]