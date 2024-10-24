import json
from google.cloud import pubsub_v1

# Initialize a Publisher and Subscriber client
publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()

# Replace with your project ID and topic/subscription IDs
project_id = "your-project-id"
topic_id = "your-topic-id"
subscription_id = "your-subscription-id"

topic_path = publisher.topic_path(project_id, topic_id)
subscription_path = subscriber.subscription_path(project_id, subscription_id)

def process_job(message):
    """Processes a Pub/Sub message."""
    job_data = json.loads(message.data.decode("utf-8")) 
    print(f"Processing job: {job_data}")
    # ... your processing logic here ...
    message.ack()  # Acknowledge the message

def receive_messages():
    """Receives messages from the Pub/Sub subscription."""
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=process_job)
    print(f"Listening for messages on {subscription_path}..\n")

    # Wrap subscriber in a 'with' block to automatically call close() when done.
    with subscriber:
        try:
            # When `timeout` is not set, result() will block indefinitely,
            # unless an exception is encountered first.
            streaming_pull_future.result()
        except TimeoutError:
            streaming_pull_future.cancel()  # Trigger the shutdown.
            streaming_pull_future.result()  # Block until the shutdown is complete.

if __name__ == "__main__":
    receive_messages()