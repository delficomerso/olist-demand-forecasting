"""Load the raw Olist CSV files into PostgreSQL, one table per file."""

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

DB_URL = "postgresql+psycopg2://olist:olist@localhost:5432/olist"
RAW_DIR = Path("data/raw")


def table_name(csv_path: Path) -> str:
    # olist_orders_dataset.csv -> orders
    name = csv_path.stem.removeprefix("olist_").removesuffix("_dataset")
    return name


def main() -> None:
    engine = create_engine(DB_URL)
    for csv_path in sorted(RAW_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path)
        name = table_name(csv_path)
        df.to_sql(name, engine, if_exists="replace", index=False)
        print(f"{name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
