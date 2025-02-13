import json
import logging
import math
from dataclasses import asdict
from datetime import datetime

import redis
from celery import Celery

from cehrgpt.generation.encoder_decoder.instruct_cehrpgt_query import (
    parse_question_to_cehrgpt_query,
)

from .config import Config
from .model_utils import load_test_patient, prompt_model

# Initialize Redis for caching
redis_client = redis.Redis(host="localhost", port=6379, db=0)
CACHE_EXPIRATION = 60 * 60 * 24  # 24 hours
logger = logging.getLogger(__name__)

# Initialize Celery
celery = Celery(
    "cehrgpt",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",  # Using database 1 for results
    include=["cehrgpt.chat.tasks"],
)
config = Config()


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


@celery.task(bind=True, name="cehrgpt.chat.tasks.generate_batch_patients")
def generate_batch_patients(self, user_input, user_session_id):
    """
    Generate a batch of synthetic patients.

    Updates progress periodically and caches results.
    """
    try:
        synthetic_patients = []
        # Create cache key
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_key = f"batch_{user_session_id}_{timestamp}"
        # Update status to started
        self.update_state(state="PROGRESS", meta={"progress": 0})

        if config.DEV_MODE:
            synthetic_patients.extend([load_test_patient()] * 128)
            self.update_state(state="PROGRESS", meta={"progress": 1.0})
            status = "SUCCESS"
        else:
            query_tuple = parse_question_to_cehrgpt_query(user_input)

            if query_tuple:
                query, n_patients = query_tuple
                self.update_state(
                    state="PROGRESS", meta={"progress": 0, "query": query}
                )
                # Your existing patient generation logic here
                # This is where you'd call your model and generate patients
                # For example:
                # patients = generate_synthetic_patients(query, n=128)
                # Simulate progress updates (replace with actual generation progress)
                batch_size = 4
                for i in range(math.ceil(n_patients / batch_size)):
                    synthetic_patients.extend(prompt_model(query, batch_size))
                    # Update progress every 10 patients
                    if i % 10 == 0:
                        progress = (i * batch_size / n_patients) * 100
                        self.update_state(
                            state="PROGRESS",
                            meta={"progress": progress, "query": query},
                        )
                status = "SUCCESS"
            else:
                status = "FAILURE"

        # Cache results with metadata
        result_data = {
            "query": user_input,
            "timestamp": timestamp,
            "user_session_id": user_session_id,
            "patients": [
                asdict(patient) for patient in synthetic_patients
            ],  # Replace with actual patient data
            "status": status,
            "total_generated": len(synthetic_patients),
        }

        json_data = json.dumps(result_data, cls=DateTimeEncoder)
        redis_client.setex(cache_key, CACHE_EXPIRATION, json_data)

        return {
            "status": "completed",
            "cache_key": cache_key,
            "result_url": f"/results/{cache_key}",
        }

    except Exception as e:
        logger.error(f"Task failed: {str(e)}", exc_info=True)
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
