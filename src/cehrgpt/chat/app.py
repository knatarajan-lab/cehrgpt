import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime

from celery.exceptions import CeleryError
from flask import Flask, jsonify, render_template, request, session, url_for

from .model_utils import handle_query, load_cehrgpt_patient_from_json
from .tasks import generate_batch_patients, redis_client

logger = logging.getLogger(__name__)


app = Flask(__name__)
app.secret_key = "your-secret-key-here"  # Replace with a real secret key

# Store user sessions and their data
user_sessions = {}


class UserSession:
    def __init__(self):
        self.synthetic_patients = []  # List to store CehrGptPatient objects
        self.last_accessed = datetime.now()
        self.conversation_history = []  # List to store conversation messages

    def add_message(self, role, content, is_patient_data=False):
        """Add a message to the conversation history.

        Args:
            role: 'user' or 'assistant'
            content: message content or patient data dict
            is_patient_data: boolean indicating if content is patient data
        """
        self.conversation_history.append(
            {
                "role": role,
                "content": content,
                "is_patient_data": is_patient_data,
                "timestamp": datetime.now().isoformat(),
            }
        )


def get_or_create_session():
    """Get existing session or create new one."""
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())

    user_id = session["user_id"]
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession()

    user_sessions[user_id].last_accessed = datetime.now()
    return user_sessions[user_id]


@app.route("/")
def index():
    user_session = get_or_create_session()
    # Pass the conversation history to the template
    return render_template(
        "index.html", conversation_history=user_session.conversation_history
    )


@app.route("/send", methods=["POST"])
def send():
    user_session = get_or_create_session()
    user_query = request.json["query"]

    # Store user message
    user_session.add_message("user", user_query)
    try:
        # Get response from your model
        query_response = handle_query(user_query)
        # If response contains patient data, store it in the session
        if isinstance(query_response, dict) and "visits" in query_response:
            user_session.synthetic_patients = [query_response]
            # Store formatted patient data response
            user_session.add_message("assistant", query_response, is_patient_data=True)
        else:
            # Store regular message response
            user_session.add_message(
                "assistant",
                query_response.get("message", str(query_response)),
                is_patient_data=False,
            )
        return jsonify(query_response)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        user_session.add_message("assistant", error_message)
        return jsonify({"message": error_message}), 500


@app.route("/conversation")
def get_conversation():
    """Endpoint to fetch conversation history."""
    user_session = get_or_create_session()
    return jsonify(user_session.conversation_history)


@app.route("/batch", methods=["POST"])
def start_batch():
    user_query = request.json.get("query")
    user_session = get_or_create_session()
    user_session_id = session.get("user_id")
    # Start Celery task
    task = generate_batch_patients.delay(user_query, user_session_id)
    system_message = (
        "Batch generation started. Click here to "
        f'<a href="{url_for("render_task_results", task_id=task.id)}" '
        'target="_blank">View Results</a>'
    )

    # Store user message
    user_session.add_message("user", user_query)
    user_session.add_message("assistant", system_message)
    return jsonify(
        {
            "task_id": task.id,
            "status_url": url_for("render_task_results", task_id=task.id),
            "message": system_message,
        }
    )


@app.route("/status/<task_id>")
def task_status(task_id):
    try:
        task = generate_batch_patients.AsyncResult(task_id)

        # Check if task exists
        if not task:
            return jsonify({"state": "FAILURE", "error": "Task not found"}), 404

        # Handle different task states
        if task.state == "PENDING":
            response = {
                "state": task.state,
                "progress": 0,
            }
        elif task.state == "FAILURE":
            # Safe error extraction
            error_info = (
                str(task.info)
                if task.info
                else "An error occurred during task execution"
            )
            response = {"state": task.state, "error": error_info}
        elif task.state == "SUCCESS":
            try:
                result = task.get()
                cache_key = result.get("cache_key")

                # Get data from Redis
                data = redis_client.get(cache_key)
                if data:
                    try:
                        result_data = json.loads(data)
                        return jsonify(
                            {
                                "state": "SUCCESS",
                                "progress": 100,
                                "total_generated": result_data.get(
                                    "total_generated", 0
                                ),
                                "result_url": result.get("result_url"),
                            }
                        )
                    except json.JSONDecodeError:
                        return jsonify(
                            {
                                "state": "FAILURE",
                                "error": "Invalid data format in cache",
                            }
                        )
                else:
                    return jsonify(
                        {
                            "state": "FAILURE",
                            "error": "Results no longer available in cache",
                        }
                    )
            except Exception as e:
                return jsonify(
                    {
                        "state": "FAILURE",
                        "error": f"Error retrieving task results: {str(e)}",
                    }
                )
        else:  # state = 'PROGRESS' or other states
            try:
                response = {
                    "state": task.state,
                    "progress": task.info.get("progress", 0) if task.info else 0,
                }
            except Exception as e:
                response = {
                    "state": task.state,
                    "progress": 0,
                    "error": f"Error getting progress: {str(e)}",
                }

        return jsonify(response)

    except ValueError:
        # Handle the specific ValueError from Celery's exception_to_python
        return (
            jsonify(
                {
                    "state": "FAILURE",
                    "error": "Task failed with invalid exception information",
                }
            ),
            500,
        )
    except Exception as e:
        return (
            jsonify(
                {"state": "FAILURE", "error": f"Error checking task status: {str(e)}"}
            ),
            500,
        )


@app.route("/results/<cache_key>")
def get_results(cache_key):
    data = redis_client.get(cache_key)
    if data:
        result_data = json.loads(data)
        return jsonify(
            {
                "query": result_data["query"],
                "timestamp": result_data["timestamp"],
                "total_generated": len(result_data["patients"]),
                "patients": result_data["patients"],
            }
        )
    return jsonify({"error": "Results not found"}), 404


@app.route("/task/<task_id>")
def render_task_results(task_id):
    return render_template("task_results.html")


@app.route("/api/patient-stats/<task_id>")
def get_patient_stats(task_id):
    stats = {
        "demographics": {
            "gender": defaultdict(int),
            "race": defaultdict(int),
            "age_groups": defaultdict(int),
        },
        "visits": {
            "types": defaultdict(int),
            "events_by_domain": defaultdict(int),
            "visits_by_month": defaultdict(int),
        },
    }

    try:
        # Log the task ID being processed
        logger.info(f"Processing stats for task ID: {task_id}")

        try:
            task = generate_batch_patients.AsyncResult(task_id)
        except CeleryError as ce:
            logger.error(f"Celery error retrieving task: {ce}")
            return jsonify({"error": "Failed to retrieve task status"}), 500

        logger.info(f"Task state: {task.state}")

        if task.state == "SUCCESS":
            try:
                result = task.get()
            except Exception as e:
                logger.error(f"Error getting task result: {e}")
                return jsonify({"error": "Failed to get task result"}), 500

            cache_key = result.get("cache_key")
            if not cache_key:
                logger.error("No cache key in result")
                return jsonify({"error": "No cache key found"}), 404

            try:
                data = redis_client.get(cache_key)
                if not data:
                    logger.error(f"No data found in Redis for key: {cache_key}")
                    return jsonify({"error": "No data found in cache"}), 404
                result_data = json.loads(data)
            except json.JSONDecodeError as je:
                logger.error(f"JSON decode error: {je}")
                return jsonify({"error": "Invalid data format in cache"}), 500
            except Exception as e:
                logger.error(f"Redis error: {e}")
                return jsonify({"error": "Failed to retrieve data from cache"}), 500

            patients = result_data.get("patients", [])
            if not patients:
                logger.error("No patients in result data")
                return jsonify({"error": "No patients found"}), 404

            logger.info(f"Processing {len(patients)} patients")
            current_year = datetime.now().year

            for i, patient_data in enumerate(patients):
                try:
                    patient = load_cehrgpt_patient_from_json(patient_data)

                    # Demographics
                    stats["demographics"]["gender"][patient.gender] += 1
                    stats["demographics"]["race"][patient.race] += 1

                    birth_datetime = patient.birth_datetime
                    age = current_year - birth_datetime.year
                    age_group = f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"
                    stats["demographics"]["age_groups"][age_group] += 1

                    # Process visits
                    for visit in patient.visits:
                        stats["visits"]["types"][visit.visit_type] += 1

                        for event in visit.events:
                            stats["visits"]["events_by_domain"][event.domain] += 1

                        visit_datetime = visit.visit_start_datetime
                        month_year = visit_datetime.strftime("%Y-%m")
                        stats["visits"]["visits_by_month"][month_year] += 1

                except Exception as e:
                    logger.error(f"Error processing patient {i}: {e}")
                    logger.error(f"Patient data: {patient_data}")
                    continue

            logger.info("Successfully processed all patients")

        else:
            logger.error(f"Task in invalid state: {task.state}")
            return jsonify({"error": f"Task is in {task.state} state"}), 400

    except Exception as e:
        logger.error(f"Unexpected error in get_patient_stats: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    final_stats = {
        "demographics": {
            "gender": dict(stats["demographics"]["gender"]),
            "race": dict(stats["demographics"]["race"]),
            "age_groups": dict(stats["demographics"]["age_groups"]),
        },
        "visits": {
            "types": dict(stats["visits"]["types"]),
            "events_by_domain": dict(stats["visits"]["events_by_domain"]),
            "visits_by_month": dict(sorted(stats["visits"]["visits_by_month"].items())),
        },
    }

    logger.info("Successfully generated stats")
    return jsonify(final_stats)


# Clean up old sessions periodically
@app.before_request
def cleanup_old_sessions():
    current_time = datetime.now()
    sessions_to_remove = []

    for user_id, user_session in user_sessions.items():
        # Remove sessions older than 1 hour
        if (current_time - user_session.last_accessed).total_seconds() > 3600:
            sessions_to_remove.append(user_id)

    for user_id in sessions_to_remove:
        del user_sessions[user_id]


if __name__ == "__main__":
    app.run()
