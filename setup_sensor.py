from sqlalchemy import create_engine, text
from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD,
    MYSQL_DB, MYSQL_SSL_CA
)

# ----------------------------
# Configuration — update these as needed
# ----------------------------
SENSOR_NAME      = 'Sensor 01'
BARANGAY_ID      = 1       # 1 = Abella, Naga City
ESTABLISHMENT_ID = None    # None = not linked to an establishment

def setup_sensor():
    url = (
        f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
        f"?ssl_ca={MYSQL_SSL_CA}"
    )
    engine = create_engine(url, pool_pre_ping=True)

    with engine.connect() as conn:
        # Check if sensor already exists
        existing = conn.execute(
            text("SELECT sensor_id, sensor_name FROM sensor WHERE sensor_name = :name"),
            {"name": SENSOR_NAME}
        ).fetchone()

        if existing:
            print(f"ℹ️  Sensor already exists: sensor_id={existing[0]}, name={existing[1]}")
            print(f"👉 Use SENSOR_ID={existing[0]} in import_csv.py")
            return existing[0]

        # Insert sensor — only columns that actually exist in the table
        result = conn.execute(
            text("""
                INSERT INTO sensor (sensor_name, barangay_id, establishment_id)
                VALUES (:name, :barangay_id, :establishment_id)
            """),
            {
                "name"            : SENSOR_NAME,
                "barangay_id"     : BARANGAY_ID,
                "establishment_id": ESTABLISHMENT_ID
            }
        )
        conn.commit()
        sensor_id = result.lastrowid
        print(f"✅ Sensor inserted!")
        print(f"    sensor_id  : {sensor_id}")
        print(f"    sensor_name: {SENSOR_NAME}")
        print(f"    barangay_id: {BARANGAY_ID} (Abella, Naga City)")
        print(f"\n👉 Use SENSOR_ID={sensor_id} in import_csv.py")
        return sensor_id

if __name__ == "__main__":
    setup_sensor()