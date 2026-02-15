import sys
import logging
from train_and_predict import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("WEEKLY PIPELINE EXECUTION STARTED")
        logger.info("=" * 60)
        
        # This calls your existing function!
        run_pipeline()
        
        logger.info("=" * 60)
        logger.info("WEEKLY PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)