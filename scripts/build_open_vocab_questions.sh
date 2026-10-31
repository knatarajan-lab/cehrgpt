#!/bin/sh

# Build the questions of the patient timeline QA head (open vocabulary):
#   1. the question universe, from the OHDSI vocabulary alone (no tokenizer needed);
#   2. the embeddings of the question names, with a frozen encoder (BioLORD-2023 by default).
# Both are done once. The training cuts the questions to the vocabulary of its tokenizer when it starts.

usage() {
    echo "Usage: $0 --vocab_path=DIR --output_dir=DIR [options]"
    echo ""
    echo "Writes <output_dir>/question_universe/ and <output_dir>/question_embeddings/."
    echo ""
    echo "Options:"
    echo "  --vocab_path=DIR         OHDSI vocabulary folder with concept/ and concept_ancestor/ as parquet (required)"
    echo "  --output_dir=DIR         Where to write the universe and the embeddings (required)"
    echo "  --domains=LIST           Domains of the universe, comma or space separated"
    echo "                           (default: Condition,Procedure,Drug,Visit)"
    echo "  --max_depth=NUM          Keep the concepts at most this many levels below their category (default: all)"
    echo "  --model=NAME             Encoder of the Hugging Face hub, or a local folder (default: FremyCompany/BioLORD-2023)"
    echo "  --revision=SHA           Commit of the encoder on the hub; it is recorded and enforced at inference"
    echo "  --template=TEXT          Applied to each name, with {text} (default: {text})"
    echo "  --batch_size=NUM         Names per batch of the encoder (default: 128)"
    echo "  --device=DEVICE          Device of the encoder, e.g. cuda (default: chosen by the encoder)"
    echo "  --fp16                   Run the encoder in half precision"
    echo "  --overwrite              Build the universe again and embed the names again (default: an existing universe"
    echo "                           is kept, and the embeddings that were computed are reused)"
    echo "  --skip_universe          Only embed: the universe is there already"
    echo "  --skip_embeddings        Only build the universe"
    echo "  --python=COMMAND         The python to use (default: python)"
    echo ""
    echo "Example:"
    echo "  $0 --vocab_path=/path/to/ohdsi --output_dir=/path/to/open_vocab --device=cuda --fp16"
    exit 1
}

PYTHON="python"
DOMAINS="Condition Procedure Drug Visit"
MAX_DEPTH=""
MODEL=""
REVISION=""
TEMPLATE=""
BATCH_SIZE=""
DEVICE=""
FP16="false"
OVERWRITE="false"
SKIP_UNIVERSE="false"
SKIP_EMBEDDINGS="false"
VOCAB_PATH=""
OUTPUT_DIR=""

for arg in "$@"; do
    case "$arg" in
        --vocab_path=*) VOCAB_PATH="${arg#*=}" ;;
        --output_dir=*) OUTPUT_DIR="${arg#*=}" ;;
        --domains=*) DOMAINS=$(echo "${arg#*=}" | tr ',' ' ') ;;
        --max_depth=*) MAX_DEPTH="${arg#*=}" ;;
        --model=*) MODEL="${arg#*=}" ;;
        --revision=*) REVISION="${arg#*=}" ;;
        --template=*) TEMPLATE="${arg#*=}" ;;
        --batch_size=*) BATCH_SIZE="${arg#*=}" ;;
        --device=*) DEVICE="${arg#*=}" ;;
        --fp16) FP16="true" ;;
        --overwrite) OVERWRITE="true" ;;
        --skip_universe) SKIP_UNIVERSE="true" ;;
        --skip_embeddings) SKIP_EMBEDDINGS="true" ;;
        --python=*) PYTHON="${arg#*=}" ;;
        -h|--help) usage ;;
        *)
            echo "Unknown option: $arg"
            usage
            ;;
    esac
done

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output_dir is required"
    usage
fi
if [ "$SKIP_UNIVERSE" = "false" ] && [ -z "$VOCAB_PATH" ]; then
    echo "Error: --vocab_path is required to build the universe"
    usage
fi
if [ "$SKIP_UNIVERSE" = "true" ] && [ "$SKIP_EMBEDDINGS" = "true" ]; then
    echo "Error: nothing to do with both --skip_universe and --skip_embeddings"
    exit 1
fi
for domain in $DOMAINS; do
    case "$domain" in
        Condition|Procedure|Drug|Visit) ;;
        *)
            echo "Error: unknown domain '$domain' (use Condition, Procedure, Drug, Visit)"
            exit 1
            ;;
    esac
done

OUTPUT_DIR="${OUTPUT_DIR%/}"
UNIVERSE_DIR="$OUTPUT_DIR/question_universe"
EMBEDDINGS_DIR="$OUTPUT_DIR/question_embeddings"

# The parquet file of a domain: Condition -> condition_universe.parquet
universe_file() {
    echo "$UNIVERSE_DIR/$(echo "$1" | tr 'A-Z' 'a-z')_universe.parquet"
}

universe_complete() {
    for domain in $DOMAINS; do
        [ -f "$(universe_file "$domain")" ] || return 1
    done
    return 0
}

if [ "$SKIP_UNIVERSE" = "false" ]; then
    if [ "$OVERWRITE" = "false" ] && universe_complete; then
        echo "Step 1: the question universe is in $UNIVERSE_DIR already, keeping it (--overwrite builds it again)"
    else
        echo "Step 1: building the question universe ($DOMAINS) from $VOCAB_PATH"
        set -- --vocab_path "$VOCAB_PATH" --output_dir "$UNIVERSE_DIR" --domains $DOMAINS
        if [ -n "$MAX_DEPTH" ]; then
            set -- "$@" --max_depth "$MAX_DEPTH"
        fi
        "$PYTHON" -m cehrgpt.open_vocab.question_universe "$@" || {
            echo "Error: building the question universe failed"
            exit 1
        }
    fi
fi

if [ "$SKIP_EMBEDDINGS" = "false" ]; then
    UNIVERSE_FILES=""
    for domain in $DOMAINS; do
        file=$(universe_file "$domain")
        if [ ! -f "$file" ]; then
            echo "Error: $file does not exist; build the universe first"
            exit 1
        fi
        UNIVERSE_FILES="$UNIVERSE_FILES $file"
    done
    echo "Step 2: embedding the names of the questions into $EMBEDDINGS_DIR"
    # shellcheck disable=SC2086 -- the files are separate words
    set -- --universe_files $UNIVERSE_FILES --output_dir "$EMBEDDINGS_DIR"
    [ -n "$MODEL" ] && set -- "$@" --model "$MODEL"
    [ -n "$REVISION" ] && set -- "$@" --revision "$REVISION"
    [ -n "$TEMPLATE" ] && set -- "$@" --template "$TEMPLATE"
    [ -n "$BATCH_SIZE" ] && set -- "$@" --batch_size "$BATCH_SIZE"
    [ -n "$DEVICE" ] && set -- "$@" --device "$DEVICE"
    [ "$FP16" = "true" ] && set -- "$@" --fp16
    [ "$OVERWRITE" = "true" ] && set -- "$@" --overwrite
    "$PYTHON" -m cehrgpt.open_vocab.question_embeddings "$@" || {
        echo "Error: embedding the questions failed"
        exit 1
    }
fi

echo ""
echo "Done. In the config of the training:"
echo "  include_open_vocab: true"
echo "  open_vocab_universe_dir: \"$UNIVERSE_DIR\""
echo "  open_vocab_embeddings_dir: \"$EMBEDDINGS_DIR\""
