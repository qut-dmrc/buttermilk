# Turn JSON into CSV format

Determine your desired columns, and use JQ to extract and format:
```sh
(echo "id,outlet,url,content,source,title,answer,reasoning" && jq -r '[.id, .outlet, .url, .text, .source, .title, (.expected.answer|tostring), (.expected.reasoning|join("; "))] | @csv' tja_train.jsonl) | tee -a output.csv
```