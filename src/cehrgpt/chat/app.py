from flask import Flask, jsonify, render_template, request

from .config import Config
from .model_utils import (
    get_generation_config,
    handle_query,
    load_concept_domain_map,
    load_model,
)

app = Flask(__name__)
config = Config()
encoder_tokenizer, cehrgpt_tokenizer, model, device = load_model(config)
generation_config = get_generation_config(cehrgpt_tokenizer)
concept_name_map, concept_domain_map = load_concept_domain_map(config)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/send", methods=["POST"])
def send():
    user_input = request.json["message"]
    response = handle_query(
        user_input,
        encoder_tokenizer,
        cehrgpt_tokenizer,
        model,
        device,
        generation_config,
        concept_domain_map,
        concept_name_map,
    )
    return jsonify({"message": response})


if __name__ == "__main__":
    app.run()
