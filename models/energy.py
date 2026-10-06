from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class EnergyUsage(Base):
    __tablename__ = "energy_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    device_type: Mapped[str] = mapped_column(String(50), nullable=False)
    energy_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    cost: Mapped[float] = mapped_column(Float, nullable=False)
    usage_period: Mapped[str] = mapped_column(String(20), nullable=False)


class SolarGeneration(Base):
    __tablename__ = "solar_generation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    generation_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    weather_condition: Mapped[str] = mapped_column(String(50), nullable=False)


DATABASE_URL = "sqlite:///data/energy_data.db"

engine = create_engine(
    DATABASE_URL,
    echo=False,
)