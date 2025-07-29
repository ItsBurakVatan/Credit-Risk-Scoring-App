# data/db_connector.py
import pandas as pd
from sqlalchemy import create_engine
from config.config import DB_SETTINGS, TABLE_NAME


def get_db_engine():
    db_url = f"postgresql://{DB_SETTINGS['user']}:{DB_SETTINGS['password']}@{DB_SETTINGS['host']}:{DB_SETTINGS['port']}/{DB_SETTINGS['database']}"
    return create_engine(db_url)

def fetch_all_data():
    engine = get_db_engine()
    query = f"SELECT * FROM {TABLE_NAME};"
    df = pd.read_sql(query, engine)
    return df

def fetch_customer_by_id(engine, table_name, customer_id):
    query = f"SELECT * FROM {table_name} WHERE musteri_id = {customer_id};"
    df = pd.read_sql(query, engine)
    return df if not df.empty else None

