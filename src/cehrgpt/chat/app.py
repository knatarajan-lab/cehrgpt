import json
import uuid
from collections import defaultdict
from datetime import datetime

from flask import Flask, jsonify, render_template, request, session, url_for

from .model_utils import handle_query, load_cehrgpt_patient_from_json
from .tasks import generate_batch_patients, redis_client

app = Flask(__name__)
app.secret_key = "your-secret-key-here"  # Replace with a real secret key

# Store user sessions and their data
user_sessions = {}


class UserSession:
    def __init__(self):
        self.synthetic_patients = []  # List to store CehrGptPatient objects
        self.last_accessed = datetime.now()


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
    get_or_create_session()
    return render_template("index.html")


@app.route("/send", methods=["POST"])
def send():
    user_session = get_or_create_session()
    user_input = request.json["message"]

    try:
        # Get response from your model
        response = handle_query(user_input)

        # If response contains patient data, store it in the session
        if isinstance(response, dict) and "visits" in response:
            user_session.synthetic_patients = [
                response
            ]  # Assuming response is a single patient

        return response

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route("/batch", methods=["POST"])
def start_batch():
    query = request.json.get("query")
    user_session_id = session.get("user_id")

    # Start Celery task
    task = generate_batch_patients.delay(query, user_session_id)

    return jsonify(
        {
            "task_id": task.id,
            "status_url": url_for("task_status", task_id=task.id),
            "message": "Batch generation started. Click here to view progress: "
            f'<a href="{url_for("task_status", task_id=task.id)}" '
            'target="_blank">View Progress</a>',
        }
    )


@app.route("/status/<task_id>")
def task_status(task_id):
    task = generate_batch_patients.AsyncResult(task_id)

    # First check if the task is in a final state
    if task.state == "SUCCESS":
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
                        "total_generated": result_data.get("total_generated", 0),
                        "result_url": result.get("result_url"),
                    }
                )
            except json.JSONDecodeError:
                return jsonify(
                    {"state": "FAILURE", "error": "Invalid data format in cache"}
                )
        else:
            # Data not found in Redis despite successful task
            return jsonify(
                {"state": "FAILURE", "error": "Results no longer available in cache"}
            )

    elif task.state == "PENDING":
        response = {
            "state": task.state,
            "progress": 0,
        }
    elif task.state == "FAILURE":
        response = {"state": task.state, "error": str(task.info.get("error", ""))}
    else:  # state = 'PROGRESS'
        response = {"state": task.state, "progress": task.info.get("progress", 0)}

    return jsonify(response)


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


from .model_utils import load_cehrgpt_patient_from_json  # Make sure to import this


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
        task = generate_batch_patients.AsyncResult(task_id)
        if task.state == "SUCCESS":
            result = task.get()
            cache_key = result.get("cache_key")

            if not cache_key:
                return jsonify({"error": "No cache key found"}), 404

            data = redis_client.get(cache_key)
            if not data:
                return jsonify({"error": "No data found in cache"}), 404

            result_data = json.loads(data)
            patients = result_data.get("patients", [])

            if not patients:
                return jsonify({"error": "No patients found"}), 404

            current_year = datetime.now().year

            for patient_data in patients:
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
                    # Log the error but continue processing other patients
                    print(f"Error processing patient: {e}")
                    continue

        else:
            return jsonify({"error": f"Task is in {task.state} state"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(
        {
            "demographics": {
                "gender": dict(stats["demographics"]["gender"]),
                "race": dict(stats["demographics"]["race"]),
                "age_groups": dict(stats["demographics"]["age_groups"]),
            },
            "visits": {
                "types": dict(stats["visits"]["types"]),
                "events_by_domain": dict(stats["visits"]["events_by_domain"]),
                "visits_by_month": dict(
                    sorted(stats["visits"]["visits_by_month"].items())
                ),
            },
        }
    )


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
