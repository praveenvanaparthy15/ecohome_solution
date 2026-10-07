from __future__ import annotations
import math
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    Get a deterministic dynamic mock weather forecast for a requested date.

    Weather values are calculated from the requested date using a seasonal
    sine-wave pattern rather than a static lookup table.

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

    day_number = requested_date.timetuple().tm_yday

    # Seasonal weight based on day-of-year.
    seasonal_weight = math.sin(
        2 * math.pi * (day_number - 80) / 365
    )

    # Dynamic temperature around a base value.
    temperature = round(
        28 + (6 * seasonal_weight),
        1,
    )

    # Dynamic cloud cover between approximately 15% and 70%.
    cloud_cover = round(
        42.5 - (27.5 * seasonal_weight)
    )

    cloud_cover = max(
        10,
        min(80, cloud_cover),
    )

    # Derive condition from calculated cloud cover.
    if cloud_cover < 25:
        condition = "sunny"
    elif cloud_cover < 50:
        condition = "partly_cloudy"
    else:
        condition = "cloudy"

    # Solar generation decreases as cloud cover increases.
    solar_factor = round(
        max(
            0.45,
            min(
                1.0,
                1.0 - (cloud_cover / 100) * 0.65,
            ),
        ),
        2,
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

    Prices vary deterministically based on the requested date while
    preserving the same time-of-use structure.

    Args:
        date: Date in YYYY-MM-DD format.

    Returns:
        Hourly electricity pricing periods and rates.
    """

    try:
        requested_date = datetime.strptime(
            date,
            "%Y-%m-%d",
        ).date()
    except ValueError:
        return {
            "success": False,
            "error": "Invalid date. Use YYYY-MM-DD format.",
        }

    day_number = requested_date.timetuple().tm_yday

    # Date-derived multiplier so different dates produce different prices.
    date_multiplier = 1 + (0.10 * math.sin(2 * math.pi * (day_number - 80) / 365))

    base_rates = {
        "off_peak": 0.08,
        "standard": 0.14,
        "peak": 0.25,
    }

    adjusted_rates = {
        period: round(
            rate * date_multiplier,
            3,
        )
        for period, rate in base_rates.items()
    }

    prices = []

    for hour in range(24):

        if 0 <= hour < 6:
            period = "off_peak"

        elif 6 <= hour < 17:
            period = "standard"

        elif 17 <= hour < 21:
            period = "peak"

        else:
            period = "standard"

        prices.append(
            {
                "hour": hour,
                "period": period,
                "price_per_kwh": adjusted_rates[period],
            }
        )

    result = {
        "success": True,
        "date": date,
        "currency": "USD",
        "prices": prices,
        "off_peak_rate": adjusted_rates["off_peak"],
        "standard_rate": adjusted_rates["standard"],
        "peak_rate": adjusted_rates["peak"],
        "date_multiplier": round(date_multiplier, 4),
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
# Recent Energy Summary Tool
# ---------------------------------------------------------------------------

@tool
def get_recent_energy_summary(
    days: int = 7,
) -> dict[str, Any]:
    """
    Provide a combined summary of recent energy consumption and solar
    generation.

    Args:
        days: Number of recent days to summarize.

    Returns:
        Combined energy usage, cost, solar generation, and net energy summary.
    """

    try:
        days = max(1, min(int(days), 30))
    except (TypeError, ValueError):
        days = 7

    usage_result = get_energy_usage.invoke(
        {"days": days}
    )

    solar_result = get_solar_generation.invoke(
        {"days": days}
    )

    if not usage_result.get("success"):
        return usage_result

    if not solar_result.get("success"):
        return solar_result

    total_usage = _safe_float(
        usage_result.get("total_energy_kwh")
    )

    total_solar = _safe_float(
        solar_result.get("total_generation_kwh")
    )

    net_grid_energy = max(
        0.0,
        total_usage - total_solar,
    )

    result = {
        "success": True,
        "days": days,
        "total_energy_usage_kwh": round(
            total_usage,
            2,
        ),
        "total_solar_generation_kwh": round(
            total_solar,
            2,
        ),
        "estimated_grid_energy_kwh": round(
            net_grid_energy,
            2,
        ),
        "total_energy_cost": round(
            _safe_float(
                usage_result.get("total_cost")
            ),
            2,
        ),
        "average_daily_energy_kwh": usage_result.get(
            "average_daily_energy_kwh",
            0,
        ),
        "average_daily_cost": usage_result.get(
            "average_daily_cost",
            0,
        ),
        "record_count": usage_result.get(
            "record_count",
            0,
        ),
        "solar_record_count": solar_result.get(
            "record_count",
            0,
        ),
    }

    _log_tool(
        "get_recent_energy_summary",
        result,
    )

    return result


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

    If the persistent vector store does not exist, it is initialized and
    populated from the project's knowledge-base documents.

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

        # ---------------------------------------------------------------
        # Initialize vector store if it does not already exist.
        # ---------------------------------------------------------------

        chroma_exists = (
            VECTORSTORE_DIR.exists()
            and any(VECTORSTORE_DIR.iterdir())
        )

        if not chroma_exists:

            logger.info(
                "ChromaDB vector store not found. Initializing..."
            )

            from langchain_core.documents import Document

            knowledge_dir = PROJECT_ROOT / "data" / "documents"

            documents = []

            if knowledge_dir.exists():

                for file_path in knowledge_dir.glob("**/*"):

                    if file_path.is_file():

                        try:
                            content = file_path.read_text(
                                encoding="utf-8"
                            )

                            if content.strip():
                                documents.append(
                                    Document(
                                        page_content=content,
                                        metadata={
                                            "source": str(file_path)
                                        },
                                    )
                                )

                        except Exception:
                            logger.exception(
                                "Unable to read knowledge file: %s",
                                file_path,
                            )

            # Fallback knowledge document so the vector store can still
            # initialize if the knowledge directory is empty.
            if not documents:
                documents = [
                    Document(
                        page_content=(
                            "Run high-energy appliances such as washing "
                            "machines, dishwashers, and EV chargers during "
                            "off-peak electricity periods. Use solar "
                            "generation during daylight hours when possible. "
                            "Maintain HVAC filters and use programmable "
                            "thermostats to reduce unnecessary energy use."
                        ),
                        metadata={
                            "source": "default_ecohome_energy_tips"
                        },
                    )
                ]

            VECTORSTORE_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
            )

            split_documents = splitter.split_documents(documents)

            vectorstore = Chroma.from_documents(
                documents=split_documents,
                embedding=embeddings,
                collection_name="ecohome_energy_tips",
                persist_directory=str(VECTORSTORE_DIR),
            )

            logger.info(
                "ChromaDB initialized with %d documents.",
                len(documents),
            )

        else:

            logger.info(
                "Existing ChromaDB vector store found. Loading..."
            )

            vectorstore = Chroma(
                collection_name="ecohome_energy_tips",
                persist_directory=str(VECTORSTORE_DIR),
                embedding_function=embeddings,
            )

        # ---------------------------------------------------------------
        # Search the vector store.
        # ---------------------------------------------------------------

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
    get_recent_energy_summary,
    search_energy_tips,
    calculate_savings,
]