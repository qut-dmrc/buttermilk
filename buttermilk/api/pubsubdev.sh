#!/bin/sh

# Start a development pubsub emulator
gcloud beta emulators pubsub start --host-port=localhost:8085
export PUBSUB_EMULATOR_HOST=localhost:8085

gcloud pubsub topics create your-topic-id --project=dmrc-platforms --emulator-host=localhost:8085 
gcloud pubsub subscriptions create your-subscription-id --topic=flow --project=dmrc-platforms --emulator-host=localhost:8085

#gcloud pubsub topics publish your-topic-id --message='{"key": "value"}' --project=your-project-id --emulator-host=localhost:8085

exit 0
gcloud pubsub topics publish flow --project=dmrc-platforms --message='{"task": "format", "filename": "gs://dmrc-platforms/data/osb/FB-U2HHA647.md", "model": "gemini15pro", "template": "format_osb"}'

gcloud pubsub topics publish flow --project=dmrc-platforms --message='{"task": "format", "filename": "gs://dmrc-platforms/data/osb/FB-U2HHA647.md", "model": "gpt4o", "template": "format_osb"}'

# {"task": "format", "filename": "gs://dmrc-platforms/data/osb/FB-U2HHA647.md", "model": "gemini15pro", "template": "format_osb" }