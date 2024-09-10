# Save flow and evaluation results to BigQuery

from logging import getLogger
from typing import Any
import pandas as pd
import pydantic

from promptflow.client import PFClient as LocalPFClient
logger = getLogger()

class SaveResultsBQ(pydantic.BaseModel):
    pflocal: LocalPFClient = pydantic.Field(default_factory=LocalPFClient)
    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def process_batch(self, runs) -> pd.DataFrame:
        df = pd.DataFrame()
        for run in runs:
            df = pd.concat([df, self.process(run_name=run)])

        return df

    def process(self, run_name: str, eval_run_name: str='', run_meta: dict = {}) -> pd.DataFrame:
        details = self.pflocal.get_details(run_name)
        if details.shape[0] == 0 or 'inputs.line_number' not in details.columns or details.replace("(Failed)", None).dropna(subset='inputs.line_number').shape[0] == 0:
            logger.error(f"No rows processed in run {run_name}.")
            return pd.DataFrame()

        details = details.rename(columns={"inputs.line_number":"line_number"})


        details.loc[:, "run_name"] = run_name
        #details.loc[:, "timestamp"] = run_time

        # stack inputs into dicts
        cols = [c for c in details.columns if c.lower().startswith("inputs.")]
        df_tmp = details[cols]
        df_tmp.columns = [c.replace("inputs.", "") for c in cols]
        details.loc[:, "inputs"] = details[cols].apply(dict, axis=1)
        details = details.drop(columns=cols)

        # for outputs, keep some results and stack any other results
        cols_to_keep = ['outputs.reasons', 'outputs.predicted',
        'outputs.labels', 'outputs.metadata', 'outputs.record_id',
        'outputs.scores', 'outputs.result']
        cols_to_stack = []
        for c in details.columns:
            if c.lower().startswith("outputs.") and c not in cols_to_keep:
                cols_to_stack.append(c)

        df_tmp = details[cols_to_stack]
        df_tmp.columns = [c.replace("outputs.", "") for c in cols_to_stack]
        details.loc[:, "moderate"] = details[cols_to_stack].apply(dict, axis=1)
        details = details.drop(columns=cols_to_stack)
        details.columns = [c.replace("outputs.", "") for c in details.columns]

        # duplicate run_info metadata for each row
        details.loc[:, "run_info"] = [run_meta for _ in range(details.shape[0])]

        evals = self.pflocal.get_details(eval_run_name)
        if evals.shape[0] == 0 or 'inputs.line_number' not in evals.columns or evals.replace("(Failed)", None).dropna(subset=['inputs.line_number']).shape[0] == 0:
            logger.error(f"No rows processed in evaluation run {eval_run_name}.")
            results = details  # no evals, just return details
        else:
            # drop evaluation inputs; we already have them
            evals = evals.rename(columns={"inputs.line_number":"line_number"})
            cols = [c for c in evals.columns if c.lower().startswith("inputs.")]
            evals = evals.drop(columns=cols)

            evals = evals.rename(
                columns={"outputs.summary": "summary", "outputs.result": "result", "inputs.line_number":"line_number"}
            )

            results = details.merge(evals, how="outer", on="line_number")


        results = results.replace("(Failed)", None).dropna(subset='line_number')
        # duplicate run_meta for each row
        results.loc[:, "run_info"] = [run_meta for _ in range(results.shape[0])]

        return results
