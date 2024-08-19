import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
from promptflow.client import PFClient as LocalPFClient
from promptflow.tracing import start_trace, trace

from flows.common import COL_PREDICTION
from datatools.chains.toxicity import TOXCLIENTS
from datatools.gcloud import GCloud
import gc

BASE_DIR = Path(__file__).absolute().parent
DATASET = "gs://dmrc-platforms/data/osb_drag_toxic_train.jsonl"

##################
##
## Moderate input data using standard off-the-shelf models
##
##################
class Scorer:
    def __init__(
        self, job_name: str, model_names: list[str], dataset: str, source_identifiers: list[str], run_local_models: bool = False):
        self.model_names = model_names
        self.dataset = dataset
        self.source_identifiers = [str.lower(s) for s in source_identifiers]
        self.run_local_models = run_local_models

        self.gc = GCloud(name="automod", job=job_name)
        self.logger = self.gc.logger

    def score(self):
        # start a trace session, and print a url for user to check trace
        start_trace(collection="automod")

        # Get a handle to workspace
        # And a local promptflow client
        localclient = LocalPFClient()

        # Get the dataset and download to a temporary location
        with NamedTemporaryFile(delete=False, suffix='.jsonl', mode='w') as f:
            dataset = f.name
        df = pd.read_json(self.dataset, orient='records', lines=True)
        if self.source_identifiers:
            df = df[df.source.str.lower().isin(self.source_identifiers)]
        df.to_json(f.name, orient='records', lines=True)
        pass

        columns = {'source': r'${data.source}',
                'record_id': r'${data.id}',
                'content': r'${data.text}',
                'id': r"${data.id}",
                'run_local_models': self.run_local_models,
                'models': self.model_names}

        # assemble variants for the judge flow
        variants = [str.lower(client) for client in self.model_names]

        runs = []
        overall_summary = []
        for i, variant in enumerate(variants):
            run_name = f"{self.gc.run_id}_{i}"
            variant_string = f"${{moderate.{variant}}}"
            run_info = dict(run_name=run_name)
            try:
                # Run the judge flow, which applies the standards
                moderate_run = localclient.run(flow="automod/flows/automod/moderate", data=dataset, name=run_name, column_mapping=columns, stream=False, variant=variant_string)
                run_info['variant'] = variant
                run_info['name'] = moderate_run.name
                run_info['moderate'] = moderate_run._to_dict(exclude_additional_info=True, exclude_debug_info=True)
                # details = localclient.get_details(moderate_run)
                pass

                # Run the second flow: evaluate, which aggregates the results
                evaluation_run = localclient.run(flow="automod/flows/automod/evaluate", data=dataset, run=moderate_run, name=f"eval_{moderate_run.name}", column_mapping={"groundtruth": "${data.expected}", COL_PREDICTION: "${run.outputs.result}"}, stream=False)
                run_info['evaluate'] = evaluation_run._to_dict(exclude_additional_info=True, exclude_debug_info=True)
                # details = localclient.get_details(evaluation_run)
                pass

                runs.append

                metrics = localclient.get_metrics(evaluation_run)
                run_info['metrics'] = metrics
                info_msg = json.dumps(metrics, indent=4)
                overall_summary.append((variant, info_msg))

            except Exception as e:
                info_msg = f"FAILED: {e}, {e.args=}"
                overall_summary.append((variant, info_msg))
            finally:
                runs.append(run_info)


        # Print the overall summary
        self.logger.info("Complete!")
        for l in overall_summary:
            self.logger.info(l)

        # Save the run info
        uri = self.gc.save(data=run_info)
        self.logger.info("Saved run info to %s", uri)



if __name__ == '__main__':
    model_names = [x.__name__ for x in TOXCLIENTS][:2]
    scorer = Scorer(job_name="ots", model_names=model_names, dataset=DATASET, source_identifiers=["drag queens", "osb"])
    scorer.score()
    pass


