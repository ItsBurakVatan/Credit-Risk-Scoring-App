# config/config.py
TARGET_COLUMN = "kredi_odeme_durumu"
MODEL_PATH = "model/saved_model.pkl"

DB_SETTINGS = {
    "host": "localhost",
    "port": "5432",
    "database": "banka_kredi_risk",
    "user": "postgres",
    "password": "241308"
}

TABLE_NAME = "musteriler"