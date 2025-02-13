from celery import Celery

celery = Celery(
    "cehrgpt",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
    include=["cehrgpt.chat.tasks"],
)

# Configure Celery
celery.conf.update(
    task_track_started=True,  # Track when tasks are started
    task_ignore_result=False,  # Make sure we're not ignoring results
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

if __name__ == "__main__":
    celery.start()
