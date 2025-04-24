import asyncio

from fastapi import FastAPI
from google.cloud import pubsub

from buttermilk.api.stream import FlowRequest, flow_stream
from buttermilk.bm import bm, logger
from buttermilk.utils.utils import load_json_flexi

INPUT_SOURCE = "api"
app = FastAPI()
flows = dict()

# gcloud pubsub topics publish TOPIC_ID --message='{"task": "summarise_osb", "uri": "gs://dmrc-platforms/data/osb/FB-515JVE4X.md", "record_id": "FB-515JVE4X"}


def callback(message):
    results = None
    try:
        data = load_json_flexi(message.data)
        task = data.pop("task")
        request = FlowRequest(**data)
        message.ack()
    except Exception as e:
        message.nack()
        logger.error(f"Error parsing Pub/Sub message: {e}")
        return

    try:
        logger.info(f"Calling flow {task} for Pub/Sub job...")

        async def process_generator():
            results = []
            async for result in flow_stream(flows[task], request):
                results.append(result)
            return results

        results = asyncio.run(
            process_generator(),
        )
        message.ack()

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        message.nack()

    logger.info("Completed Pub/Sub job.")


def start_pubsub_listener():
    # publisher = pubsub.PublisherClient()
    subscriber = pubsub.SubscriberClient()
    subscription_path = subscriber.subscription_path(
        bm.cfg.pubsub.project,
        bm.cfg.pubsub.subscription,
    )
    topic_path = subscriber.topic_path(bm.cfg.pubsub.project, bm.cfg.pubsub.topic)

    # if "dead_letter_topic_id" in bm.cfg.pubsub:
    #     dead_letter_topic_path = publisher.topic_path(
    #         bm.cfg.pubsub.project,
    #         bm.cfg.pubsub.dead_letter_topic,
    #     )
    #     logger.info(
    #         "Pub/Sub forwarding failed messages to: {dead_letter_topic_path} after {bm.cfg.pubsub.max_retries} retries.",
    #     )
    #     dead_letter_policy = {
    #         "dead_letter_topic": dead_letter_topic_path,
    #         "max_delivery_attempts": bm.cfg.pubsub.max_retries,
    #     }
    # else:
    #     dead_letter_policy = None

    # # try to create the subscription if necessary
    # try:
    #     with subscriber:
    #         request = {
    #             "name": subscription_path,
    #             "topic": topic_path,
    #             "dead_letter_policy": dead_letter_policy,
    #         }
    #         subscription = subscriber.create_subscription(request)
    # except Exception as e:
    #     logger.error(
    #         f"Unable to create pub/sub subscription {subscription_path}: {e}, {e.args=}",
    #     )

    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    logger.info(f"Listening for messages on {subscription_path} topic {topic_path}...")

    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
