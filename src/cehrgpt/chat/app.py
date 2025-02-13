import uuid
from collections import defaultdict
from datetime import datetime

from flask import Flask, jsonify, render_template, request, session

from .config import Config
from .model_utils import (
    get_generation_config,
    handle_query,
    load_concept_domain_map,
    load_model,
    load_test_patient,
)

app = Flask(__name__)
app.secret_key = "your-secret-key-here"  # Replace with a real secret key

# Initialize model and configs
config = Config()
encoder_tokenizer, cehrgpt_tokenizer, model, device = load_model(config)
generation_config = get_generation_config(cehrgpt_tokenizer)
concept_name_map, concept_domain_map = load_concept_domain_map(config)

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
        response = handle_query(
            user_input,
            encoder_tokenizer,
            cehrgpt_tokenizer,
            model,
            device,
            generation_config,
            concept_domain_map,
            concept_name_map,
            config,
        )

        # If response contains patient data, store it in the session
        if isinstance(response, dict) and "visits" in response:
            user_session.synthetic_patients = [
                response
            ]  # Assuming response is a single patient

        return response

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route("/api/patient-stats")
def get_patient_stats():
    user_session = get_or_create_session()
    user_session.synthetic_patients = [load_test_patient()]
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

    current_year = datetime.now().year

    # Process patients from user's session
    for patient in user_session.synthetic_patients:
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
