gsutil ls "gs://dmrc-platforms/data/osb/*md" | while read -r uri; do
    filename=$(basename "$uri")
    # echo "About to send:"
    # echo "{\"task\": \"summarise_osb\", \"uri\": \"$uri\", \"record_id\": \"${filename%.*}\"}"
    # echo "Press RETURN to send, or Ctrl+C to abort"
    # read
    echo gcloud pubsub topics publish flow --message=\'{\"task\": \"summarise_osb\", \"uri\": \"$uri\", \"record_id\": \"${filename%.*}\"}\'
done