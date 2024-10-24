#!/bin/sh

# Start a development pubsub emulator
gcloud beta emulators pubsub start --host-port=localhost:8085
export PUBSUB_EMULATOR_HOST=localhost:8085

gcloud pubsub topics create your-topic-id --project=your-project-id --emulator-host=localhost:8085 
gcloud pubsub subscriptions create your-subscription-id --topic=your-topic-id --project=your-project-id --emulator-host=localhost:8085

#gcloud pubsub topics publish your-topic-id --message='{"key": "value"}' --project=your-project-id --emulator-host=localhost:8085

