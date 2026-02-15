import pandas as pd
from sqlalchemy import create_engine, text
from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD,
    MYSQL_DB, MYSQL_SSL_CA
)

# ----------------------------
# Configuration
# Run setup_sensor.py first to get your SENSOR_ID
# ----------------------------
CSV_PATH  = "data/hourly_data.csv"
SENSOR_ID = 1       # ← update with sensor_id printed by setup_sensor.py
BATCH_SIZE = 500

# ----------------------------
# CO2 level classifier
# Matches ENUM in sensor_data: LOW, NORMAL, HIGH, VERY HIGH
# ----------------------------
def classify_co2(co2):
    if co2 < 400:
        return 'LOW'
    elif co2 < 600:
        return 'NORMAL'
    elif co2 < 1000:
        return 'HIGH'
    else:
        return 'VERY HIGH'

# ----------------------------
# Heat index calculator (Steadman formula)
# ----------------------------
def calc_heat_index(temp_c, humidity):
    try:
        t = temp_c
        h = humidity
        hi = (-8.78469475556
              + 1.61139411 * t
              + 2.33854883889 * h
              - 0.14611605 * t * h
              - 0.012308094 * t**2
              - 0.0164248277778 * h**2
              + 0.002211732 * t**2 * h
              + 0.00072546 * t * h**2
              - 0.000003582 * t**2 * h**2)
        return round(hi, 2)
    except Exception:
        return None

# ----------------------------
# Main import
# ----------------------------
def import_csv(csv_path):
    print(f"📂 Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])

    print(f"ℹ️  CSV shape: {df.shape}")
    print(f"ℹ️  Columns found: {list(df.columns)}")
    print(df.head(3))

    # --- Rename CSV columns to match sensor_data table ---
    df.rename(columns={
        'timestamp'           : 'recorded_at',
        'co2_ppm'             : 'co2_density',
        'temperature_celsius' : 'temperature_c',
        'humidity_percent'    : 'humidity'
    }, inplace=True)

    # --- Add computed columns ---
    df['sensor_id']    = SENSOR_ID
    df['heat_index_c'] = df.apply(
        lambda row: calc_heat_index(row['temperature_c'], row['humidity']), axis=1
    )
    df['carbon_level'] = df['co2_density'].apply(classify_co2)

    # --- minute_stamp = same as recorded_at (required by table) ---
    df['minute_stamp'] = df['recorded_at']

    # --- Keep only columns that exist in sensor_data ---
    df = df[[
        'sensor_id', 'co2_density', 'temperature_c', 'humidity',
        'heat_index_c', 'carbon_level', 'recorded_at', 'minute_stamp'
    ]]

    # --- Drop rows with nulls in critical columns ---
    before = len(df)
    df.dropna(subset=['co2_density', 'temperature_c', 'humidity'], inplace=True)
    after = len(df)
    if before != after:
        print(f"⚠️  Dropped {before - after} rows with null values.")

    print(f"✅ {after} rows ready to insert.")

    # --- Connect and insert in batches ---
    url = (
        f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
        f"?ssl_ca={MYSQL_SSL_CA}"
    )
    engine = create_engine(url, pool_pre_ping=True)

    total_inserted = 0
    total_skipped  = 0

    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i : i + BATCH_SIZE]
        rows  = batch.to_dict(orient='records')

        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    INSERT IGNORE INTO sensor_data
                        (sensor_id, co2_density, temperature_c, humidity,
                         heat_index_c, carbon_level, recorded_at, minute_stamp)
                    VALUES
                        (:sensor_id, :co2_density, :temperature_c, :humidity,
                         :heat_index_c, :carbon_level, :recorded_at, :minute_stamp)
                """),
                rows
            )
            conn.commit()

        inserted = result.rowcount
        skipped  = len(batch) - inserted
        total_inserted += inserted
        total_skipped  += skipped
        print(f"  ↳ Batch {i // BATCH_SIZE + 1}: {inserted} inserted, {skipped} skipped")

    print(f"\n✅ Done! {total_inserted} rows inserted, {total_skipped} duplicates skipped.")


if __name__ == "__main__":
    import_csv(CSV_PATH)