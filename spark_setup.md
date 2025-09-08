# Spark Environment Setup

This guide provides instructions for configuring Apache Spark for CEHR-GPT data processing tasks.

## Environment Configuration

### Basic Spark Environment

Set up the core Spark environment variables:

```bash
# Set Spark home directory
export SPARK_HOME=$(python -c "import pyspark; print(pyspark.__file__.rsplit('/', 1)[0])")

# Configure Python interpreters
export PYSPARK_PYTHON=$(python -c "import sys; print(sys.executable)")
export PYSPARK_DRIVER_PYTHON=$(python -c "import sys; print(sys.executable)")

# Update Python and system paths
export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
export PATH=$SPARK_HOME/bin:$PATH
```

### Spark Worker and Executor Configuration

Configure Spark for data processing workloads:

```bash
# Worker configuration
export SPARK_WORKER_INSTANCES="1"
export SPARK_WORKER_CORES="16"
export SPARK_EXECUTOR_CORES="2"

# Memory configuration
export SPARK_DRIVER_MEMORY="12g"
export SPARK_EXECUTOR_MEMORY="12g"

# Master configuration
export SPARK_MASTER="local[64]"
```

### Consolidated Submit Options

Create a single environment variable for all Spark submit options:

```bash
export SPARK_SUBMIT_OPTIONS="--master $SPARK_MASTER --driver-memory $SPARK_DRIVER_MEMORY --executor-memory $SPARK_EXECUTOR_MEMORY --executor-cores $SPARK_EXECUTOR_CORES --conf spark.sql.adaptive.enabled=$SPARK_CONF_spark_sql_adaptive_enabled --conf spark.sql.adaptive.coalescePartitions.enabled=$SPARK_CONF_spark_sql_adaptive_coalescePartitions_enabled --conf spark.serializer=$SPARK_CONF_spark_serializer"
```

## Usage Examples

### Using Environment Variables
```bash
# Run PySpark with configured environment
pyspark

# Submit a Spark job
spark-submit your_script.py
```

### Using Submit Options
```bash
# Submit with explicit options
spark-submit $SPARK_SUBMIT_OPTIONS your_script.py
```

## Configuration Tuning

### Memory Settings

Adjust memory allocation based on your system resources:

- **Small datasets (< 10GB)**: 4-8GB driver/executor memory
- **Medium datasets (10-100GB)**: 8-16GB driver/executor memory
- **Large datasets (> 100GB)**: 16-32GB driver/executor memory

### Core Settings

Configure cores based on your CPU resources:

- **Total cores available**: Use 80-90% for Spark workers
- **Executor cores**: Generally 2-5 cores per executor for optimal performance
- **Worker instances**: Usually 1 for single-node setups

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**: Increase `SPARK_DRIVER_MEMORY` and `SPARK_EXECUTOR_MEMORY`
2. **Slow Performance**: Enable adaptive query execution and increase parallelism
3. **Serialization Issues**: Ensure Kryo serializer is configured

### Verification

Test your configuration:

```bash
echo "Spark Environment Variables:"
echo "SPARK_HOME: $SPARK_HOME"
echo "SPARK_MASTER: $SPARK_MASTER"
echo "SPARK_DRIVER_MEMORY: $SPARK_DRIVER_MEMORY"
echo "SPARK_EXECUTOR_MEMORY: $SPARK_EXECUTOR_MEMORY"
```
