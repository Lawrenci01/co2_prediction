from sqlalchemy import create_engine, text
from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD,
    MYSQL_DB, MYSQL_SSL_CA
)

def check_db():
    url = (
        f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
        f"?ssl_ca={MYSQL_SSL_CA}"
    )
    engine = create_engine(url, pool_pre_ping=True)

    with engine.connect() as conn:
        # Show exact columns of sensor table
        print("ℹ️  sensor table columns:")
        cols = conn.execute(text("SHOW COLUMNS FROM sensor")).fetchall()
        for c in cols:
            print(f"    {dict(c._mapping)}")

        # Show exact columns of sensor_data table
        print("\nℹ️  sensor_data table columns:")
        cols = conn.execute(text("SHOW COLUMNS FROM sensor_data")).fetchall()
        for c in cols:
            print(f"    {dict(c._mapping)}")

if __name__ == "__main__":
    check_db()