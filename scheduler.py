import schedule
import time
from datetime import datetime
from train_and_predict import run_pipeline

# ----------------------------
# Schedule Configuration
# The pipeline runs automatically every week on Sunday at midnight.
# You can change the schedule below without touching train_and_predict.py.
# ----------------------------

def job():
    print(f"\n Scheduler triggered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        run_pipeline()
    except Exception as e:
        print(f" Pipeline failed: {e}")

# --- Schedule options (uncomment the one you want) ---

# Every week on Sunday at midnight
schedule.every().sunday.at("00:00").do(job)

# Every day at midnight (useful for testing with live IoT data)
# schedule.every().day.at("00:00").do(job)

# Every hour (for testing only — remove in production)
# schedule.every().hour.do(job)

# Every 5 minutes (for quick testing only)
# schedule.every(5).minutes.do(job)

# ----------------------------
if __name__ == "__main__":
    print(" Scheduler started.")
    print(f"   Next run: {schedule.next_run()}")
    print("   Press Ctrl+C to stop.\n")

    # Run once immediately on startup so you don't wait a full week
    print("  Running pipeline immediately on startup...")
    job()

    # Then keep running on schedule
    while True:
        schedule.run_pending()
        time.sleep(30)  # check every 30 seconds