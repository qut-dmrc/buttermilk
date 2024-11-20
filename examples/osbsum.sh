cat cases.txt | while read -r uri; do
    python -m buttermilk.runner.cli "+flows=[summarise_osb]" "+flow=summarise_osb" "+record.uri=$uri"
    echo "finished $uri; sleeping 20s"
    sleep 20
done