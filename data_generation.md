# CEHR-GPT Data Generation

This guide covers the process of generating pre-training data for CEHR-GPT from OMOP-formatted healthcare datasets.

## Prerequisites

Before starting data generation, ensure you have:

1. **Spark Environment**: Configured Apache Spark (see [Spark Setup README](./spark_setup.md))
2. **OMOP Data**: Healthcare data in OMOP Common Data Model format
3. **Environment Variables**: Required paths and directories set up

## Required Environment Variables

Set up the necessary directory paths:

```bash
# CEHR-GPT installation directory
export CEHRGPT_HOME=$(git rev-parse --show-toplevel)

# OMOP input data directory
export OMOP_DIR="/path/to/omop/data"

# Output directory for processed data
export CEHR_GPT_DATA_DIR="/path/to/output/data"
```

## Step 1: Configure Spark for Data Processing

Set up Spark environment variables optimized for healthcare data processing:

```bash
# Worker configuration
export SPARK_WORKER_INSTANCES="1"
export SPARK_MASTER="local[16]"
export SPARK_WORKER_CORES="16"
export SPARK_EXECUTOR_CORES="4"

# Memory configuration
export SPARK_DRIVER_MEMORY="20g"
export SPARK_EXECUTOR_MEMORY="20g"

export SPARK_SUBMIT_OPTIONS="--master $SPARK_MASTER --driver-memory $SPARK_DRIVER_MEMORY --executor-memory $SPARK_EXECUTOR_MEMORY --executor-cores $SPARK_EXECUTOR_CORES"
```

### Configuration Guidelines

**Memory Allocation:**
- **Small datasets (< 1M patients)**: 8GB driver/executor memory
- **Medium datasets (1-10M patients)**: 12-16GB driver/executor memory
- **Large datasets (> 10M patients)**: 20-32GB driver/executor memory

**Core Allocation:**
- Adjust `SPARK_WORKER_CORES` based on available CPU cores
- Keep `SPARK_EXECUTOR_CORES` at 2-4 for optimal performance
- Reserve 2-4 cores for system processes

## Step 2: Generate Pre-training Data

Execute the data generation script:

```bash
sh $CEHRGPT_HOME/scripts/create_cehrgpt_pretraining_data.sh \
  --input_folder $OMOP_DIR \
  --output_folder $CEHR_GPT_DATA_DIR \
  --start_date "1985-01-01"
```

### Script Parameters

- `--input_folder`: Directory containing OMOP-formatted data files
- `--output_folder`: Directory where processed data will be saved
- `--start_date`: Earliest date for including patient records (format: YYYY-MM-DD)

## Performance Optimization

### For Large Datasets

```bash
# Increase parallelism
export SPARK_SQL_SHUFFLE_PARTITIONS="800"

# Enable dynamic allocation
export SPARK_CONF_spark_dynamicAllocation_enabled="true"
export SPARK_CONF_spark_dynamicAllocation_minExecutors="2"
export SPARK_CONF_spark_dynamicAllocation_maxExecutors="20"
```

### Memory Optimization

```bash
# Tune garbage collection
export SPARK_CONF_spark_executor_extraJavaOptions="-XX:+UseG1GC -XX:+PrintGCDetails"

# Optimize serialization
export SPARK_CONF_spark_serializer_objectStreamReset="100"
```

## Troubleshooting

### Common Issues

**Out of Memory Errors:**
```bash
# Increase driver memory
export SPARK_DRIVER_MEMORY="20g"

# Increase executor memory
export SPARK_EXECUTOR_MEMORY="16g"
```

**Slow Performance:**
```bash
# Increase parallelism
export SPARK_WORKER_CORES="32"

# Enable adaptive query execution
export SPARK_CONF_spark_sql_adaptive_enabled="true"
```
