import multiprocessing

from celery import Celery

# Force the start method to be 'spawn' for CUDA compatibility
try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass  # Already set

celery = Celery(
    "cehrgpt",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
    include=["cehrgpt.chat.tasks"],
)

# Combine all Celery configurations
celery.conf.update(
    # Application settings
    task_track_started=True,
    task_ignore_result=False,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Worker settings for CUDA compatibility
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1,  # Create new worker for each task
    task_acks_late=True,
    worker_cancel_long_running_tasks_on_connection_loss=True,
    worker_concurrency=1,  # Don't use multiple processes for GPU tasks
)

if __name__ == "__main__":
    celery.start()
