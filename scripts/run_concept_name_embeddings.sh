#!/bin/sh

# Embed OMOP concept names (SNOMED conditions, RxNorm ingredients) with a text encoder and
# check how well the vectors separate disease groups / ATC classes.
#
# See src/cehrgpt/analysis/concept_name_embeddings.py for what each run produces.

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Where the vocabulary is (give --vocab_dir, or the tables one by one):"
    echo "  --vocab_dir=DIR                Folder with the OMOP vocabulary tables. For each table the first"
    echo "                                 of these is used: <table>/ (parquet folder), <table>.parquet,"
    echo "                                 <TABLE>.csv (Athena, tab separated), <table>.csv"
    echo "  --concept_path=PATH            The concept table (overrides --vocab_dir)"
    echo "  --concept_ancestor_path=PATH   The concept_ancestor table (overrides --vocab_dir)"
    echo "  --concept_relationship_path=PATH  The concept_relationship table (overrides --vocab_dir)"
    echo "  --output_dir=DIR               Where everything is written (required)"
    echo ""
    echo "What to run:"
    echo "  --analyses=LIST                snomed, rxnorm or both, space separated (default: \"snomed rxnorm\")"
    echo "  --snomed_domains=LIST          Domains of the SNOMED concepts to embed, space separated, or 'all'"
    echo "                                 (default: Condition, the domain the disease groups apply to)"
    echo "  --snomed_group_depths=LIST     Levels below the root at which the disease groups are taken,"
    echo "                                 one run per value (default: \"2\"; 1 = the root's children)"
    echo "  --snomed_group_root=NAME       Exact name of the root concept (default: Disease)"
    echo "  --snomed_group_root_id=ID      Use this concept id as the root instead of looking up the name"
    echo "  --atc_levels=LIST              ATC levels for the RxNorm groups, one run per value"
    echo "                                 (default: \"1 2\"; 1 = anatomical main group, 2 = therapeutic subgroup)"
    echo "  --rxnorm_concept_classes=LIST  Comma separated concept classes of the RxNorm concepts, or 'all'"
    echo "                                 (default: Ingredient)"
    echo ""
    echo "Encoder:"
    echo "  --model=NAME_OR_PATH           Hugging Face encoder (default: FremyCompany/BioLORD-2023; its"
    echo "                                 license requires UMLS and SNOMED CT licences)"
    echo "  --batch_size=NUM               Encoder batch size (default: 256)"
    echo "  --fp16                         Run the encoder in half precision on a GPU"
    echo "  --gpu_id=ID                    Use this GPU only (sets CUDA_VISIBLE_DEVICES; default: whatever is visible)"
    echo "  --text_template=TEMPLATE       Text that is embedded, e.g. '{domain_id}: {concept_name}'"
    echo "                                 (default: '{concept_name}')"
    echo ""
    echo "Other:"
    echo "  --max_concepts=NUM             Embed a random sample of this many concepts per analysis (default: all)"
    echo "  --max_plot_points=NUM          Points drawn in the projection (default: 20000)"
    echo "  --projection=METHOD            auto, tsne, umap or pca (default: auto)"
    echo "  --overwrite                    Recompute the embeddings even if a matching cache exists"
    echo "  --extra_args=ARGS              Passed through to the Python tool as they are, e.g."
    echo "                                 '--min_group_size 5 --knn 20'"
    echo "  --use_installed                Use the installed cehrgpt instead of the source next to this script"
    echo "  --python=CMD                   Python to use (default: python)"
    echo "  --dry_run                      Print the commands and do nothing"
    echo "  -h, --help                     Show this help"
    echo ""
    echo "Example:"
    echo "  nohup sh $0 --vocab_dir=/data/athena --output_dir=/data/concept_embeddings --gpu_id=0 &"
}

# Default values
VOCAB_DIR=""
CONCEPT_PATH=""
CONCEPT_ANCESTOR_PATH=""
CONCEPT_RELATIONSHIP_PATH=""
OUTPUT_DIR=""
ANALYSES="snomed rxnorm"
SNOMED_DOMAINS="Condition"
SNOMED_GROUP_DEPTHS="2"
SNOMED_GROUP_ROOT="Disease"
SNOMED_GROUP_ROOT_ID=""
ATC_LEVELS="1 2"
RXNORM_CONCEPT_CLASSES="Ingredient"
MODEL="FremyCompany/BioLORD-2023"
BATCH_SIZE="256"
FP16="false"
GPU_ID=""
TEXT_TEMPLATE="{concept_name}"
MAX_CONCEPTS=""
MAX_PLOT_POINTS="20000"
PROJECTION="auto"
OVERWRITE="false"
EXTRA_ARGS=""
USE_INSTALLED="false"
PYTHON="python"
DRY_RUN="false"

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --vocab_dir=*)
            VOCAB_DIR="${arg#*=}"
            ;;
        --concept_path=*)
            CONCEPT_PATH="${arg#*=}"
            ;;
        --concept_ancestor_path=*)
            CONCEPT_ANCESTOR_PATH="${arg#*=}"
            ;;
        --concept_relationship_path=*)
            CONCEPT_RELATIONSHIP_PATH="${arg#*=}"
            ;;
        --output_dir=*)
            OUTPUT_DIR="${arg#*=}"
            ;;
        --analyses=*)
            ANALYSES="${arg#*=}"
            ;;
        --snomed_domains=*)
            SNOMED_DOMAINS="${arg#*=}"
            ;;
        --snomed_group_depths=*)
            SNOMED_GROUP_DEPTHS="${arg#*=}"
            ;;
        --snomed_group_root=*)
            SNOMED_GROUP_ROOT="${arg#*=}"
            ;;
        --snomed_group_root_id=*)
            SNOMED_GROUP_ROOT_ID="${arg#*=}"
            ;;
        --atc_levels=*)
            ATC_LEVELS="${arg#*=}"
            ;;
        --rxnorm_concept_classes=*)
            RXNORM_CONCEPT_CLASSES="${arg#*=}"
            ;;
        --model=*)
            MODEL="${arg#*=}"
            ;;
        --batch_size=*)
            BATCH_SIZE="${arg#*=}"
            ;;
        --fp16)
            FP16="true"
            ;;
        --gpu_id=*)
            GPU_ID="${arg#*=}"
            ;;
        --text_template=*)
            TEXT_TEMPLATE="${arg#*=}"
            ;;
        --max_concepts=*)
            MAX_CONCEPTS="${arg#*=}"
            ;;
        --max_plot_points=*)
            MAX_PLOT_POINTS="${arg#*=}"
            ;;
        --projection=*)
            PROJECTION="${arg#*=}"
            ;;
        --overwrite)
            OVERWRITE="true"
            ;;
        --extra_args=*)
            EXTRA_ARGS="${arg#*=}"
            ;;
        --use_installed)
            USE_INSTALLED="true"
            ;;
        --python=*)
            PYTHON="${arg#*=}"
            ;;
        --dry_run)
            DRY_RUN="true"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            usage
            exit 1
            ;;
    esac
done

# Find a table of the vocabulary: parquet folder, parquet file, or Athena csv
find_table() {
    table="$1"
    upper=$(echo "$table" | tr '[:lower:]' '[:upper:]')
    for candidate in "$VOCAB_DIR/$table" "$VOCAB_DIR/$table.parquet" "$VOCAB_DIR/$upper.csv" "$VOCAB_DIR/$table.csv"; do
        if [ -e "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

resolve_table() {
    # $1 = current value, $2 = table name
    if [ -n "$1" ]; then
        echo "$1"
    elif [ -n "$VOCAB_DIR" ]; then
        find_table "$2" || true
    fi
}

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output_dir is required"
    usage
    exit 1
fi

CONCEPT_PATH=$(resolve_table "$CONCEPT_PATH" concept)
CONCEPT_ANCESTOR_PATH=$(resolve_table "$CONCEPT_ANCESTOR_PATH" concept_ancestor)
CONCEPT_RELATIONSHIP_PATH=$(resolve_table "$CONCEPT_RELATIONSHIP_PATH" concept_relationship)

if [ -z "$CONCEPT_PATH" ] || [ ! -e "$CONCEPT_PATH" ]; then
    echo "Error: the concept table was not found; give --vocab_dir or --concept_path"
    exit 1
fi

for analysis in $ANALYSES; do
    case $analysis in
        snomed)
            if [ -z "$CONCEPT_ANCESTOR_PATH" ] || [ ! -e "$CONCEPT_ANCESTOR_PATH" ]; then
                echo "Error: the snomed analysis needs the concept_ancestor table (--vocab_dir or --concept_ancestor_path)"
                exit 1
            fi
            ;;
        rxnorm)
            if [ -z "$CONCEPT_RELATIONSHIP_PATH" ] || [ ! -e "$CONCEPT_RELATIONSHIP_PATH" ]; then
                echo "Error: the rxnorm analysis needs the concept_relationship table (--vocab_dir or --concept_relationship_path)"
                exit 1
            fi
            if [ -z "$CONCEPT_ANCESTOR_PATH" ] || [ ! -e "$CONCEPT_ANCESTOR_PATH" ]; then
                echo "Error: the rxnorm analysis needs the concept_ancestor table to climb the ATC hierarchy"
                exit 1
            fi
            ;;
        *)
            echo "Error: unknown analysis '$analysis' (use snomed and/or rxnorm)"
            exit 1
            ;;
    esac
done

MAIN_LOG="$OUTPUT_DIR/run.log"
if [ "$DRY_RUN" = "false" ]; then
    mkdir -p "$OUTPUT_DIR/logs"
fi

log() {
    message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    if [ "$DRY_RUN" = "true" ]; then
        echo "$message"
    else
        echo "$message" | tee -a "$MAIN_LOG"
    fi
}

# The source next to this script, so a reinstall of cehrgpt is not needed
if [ "$USE_INSTALLED" = "false" ]; then
    REPO_SRC="$(cd "$(dirname "$0")/.." && pwd)/src"
    if [ ! -d "$REPO_SRC/cehrgpt" ]; then
        echo "Error: $REPO_SRC/cehrgpt not found; use --use_installed"
        exit 1
    fi
    export PYTHONPATH="$REPO_SRC${PYTHONPATH:+:$PYTHONPATH}"
fi
# transformers should not try to import TensorFlow
export USE_TF=0
if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

log "Starting the concept name embedding analyses"
log "  concept table:        $CONCEPT_PATH"
log "  concept_ancestor:     ${CONCEPT_ANCESTOR_PATH:-none}"
log "  concept_relationship: ${CONCEPT_RELATIONSHIP_PATH:-none}"
log "  model:                $MODEL"
log "  analyses:             $ANALYSES"
log "  output:               $OUTPUT_DIR"

# Run the Python tool once.
# $1 = name of the run (its output folder and log), $2 = the embedding cache folder; the
# remaining arguments are added to the command line.
run_tool() {
    name="$1"
    cache="$2"
    shift 2
    out="$OUTPUT_DIR/$name"
    run_log="$OUTPUT_DIR/logs/$name.log"
    set -- --concept_path "$CONCEPT_PATH" --output_dir "$out" --cache_dir "$cache" \
        --model "$MODEL" --batch_size "$BATCH_SIZE" --text_template "$TEXT_TEMPLATE" \
        --max_plot_points "$MAX_PLOT_POINTS" --projection "$PROJECTION" "$@"
    if [ -n "$CONCEPT_ANCESTOR_PATH" ]; then
        set -- "$@" --concept_ancestor_path "$CONCEPT_ANCESTOR_PATH"
    fi
    if [ -n "$CONCEPT_RELATIONSHIP_PATH" ]; then
        set -- "$@" --concept_relationship_path "$CONCEPT_RELATIONSHIP_PATH"
    fi
    if [ -n "$MAX_CONCEPTS" ]; then
        set -- "$@" --max_concepts "$MAX_CONCEPTS"
    fi
    if [ "$FP16" = "true" ]; then
        set -- "$@" --fp16
    fi
    if [ "$OVERWRITE" = "true" ]; then
        set -- "$@" --overwrite
    fi
    if [ -n "$EXTRA_ARGS" ]; then
        # shellcheck disable=SC2086
        set -- "$@" $EXTRA_ARGS
    fi
    log "Run '$name' -> $out (log: $run_log)"
    if [ "$DRY_RUN" = "true" ]; then
        echo "  $PYTHON -m cehrgpt.analysis.concept_name_embeddings $*"
        return 0
    fi
    mkdir -p "$out" "$cache"
    if ! "$PYTHON" -m cehrgpt.analysis.concept_name_embeddings "$@" >> "$run_log" 2>&1; then
        log "Run '$name' failed; the last lines of $run_log:"
        tail -n 20 "$run_log"
        exit 1
    fi
    summarize "$out"
}

# Print the numbers that matter from a finished run
summarize() {
    metrics="$1/metrics.json"
    if [ ! -f "$metrics" ]; then
        return 0
    fi
    "$PYTHON" - "$metrics" <<'EOF' | while IFS= read -r line; do log "  $line"; done
import json, sys
m = json.load(open(sys.argv[1]))
print(f"{m['vocabulary_id']}: {m['num_concepts']} concepts, {m['embedding_dim']}-d, projection {m.get('projection')}")
a = m.get("cluster_agreement")
if a:
    print(f"k-means ({a['n_clusters']} clusters) vs groups: ARI {a['adjusted_rand_index']:.3f}, NMI {a['normalized_mutual_information']:.3f}, homogeneity {a['homogeneity']:.3f}")
n = m.get("neighbour_group_purity")
if n:
    print(f"nearest {n['k']} neighbours in the same group: {n['purity']:.3f} (chance {n['chance']:.3f})")
p = m.get("parent_retrieval")
if p and p.get("n_queries"):
    print("parent among nearest neighbours: " + ", ".join(f"@{k.split('_')[-1]} {v:.3f}" for k, v in p.items() if k.startswith("accuracy")))
EOF
}

for analysis in $ANALYSES; do
    case $analysis in
        snomed)
            for depth in $SNOMED_GROUP_DEPTHS; do
                # one embedding cache for every depth: the concepts and the model are the same
                set --
                if [ "$SNOMED_DOMAINS" != "all" ]; then
                    set -f
                    # shellcheck disable=SC2086
                    set -- --domains $SNOMED_DOMAINS
                    set +f
                fi
                set -- "$@" --vocabulary_id SNOMED --color_by disease_group \
                    --group_root_name "$SNOMED_GROUP_ROOT" --group_depth "$depth"
                if [ -n "$SNOMED_GROUP_ROOT_ID" ]; then
                    set -- "$@" --group_root_id "$SNOMED_GROUP_ROOT_ID"
                fi
                run_tool "snomed_depth${depth}" "$OUTPUT_DIR/cache/snomed" "$@"
            done
            ;;
        rxnorm)
            for level in $ATC_LEVELS; do
                set -- --vocabulary_id RxNorm --color_by atc_group --atc_level "$level"
                if [ "$RXNORM_CONCEPT_CLASSES" != "all" ]; then
                    # the classes may contain spaces ("Clinical Drug"), so split on commas only
                    old_ifs=$IFS
                    IFS=,
                    set -f
                    # shellcheck disable=SC2086
                    set -- "$@" --concept_classes $RXNORM_CONCEPT_CLASSES
                    set +f
                    IFS=$old_ifs
                fi
                run_tool "rxnorm_atc${level}" "$OUTPUT_DIR/cache/rxnorm" "$@"
            done
            ;;
    esac
done

log "Done. Look at <output_dir>/<run>/projection.png, metrics.json and groups.csv"
