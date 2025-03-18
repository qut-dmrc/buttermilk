######
# Slurp all data we can from an api.
#
# Input:
#
#   data: a dictionary, json string, list of dicts, or dataframe with already collected data
#       OR
#   uri: the api endpoint
#   params: a dict of parameters

#
# Output:
#   Data saved neatly in a standard GCS location
#   Full logging informatoin saved to Google Cloud Logging


import httpx

from buttermilk._core.agent import SingleAgent


class Slurp(SingleAgent):
    #         endpoint = endpoint
    #         key = key
    #         secret = secret

    #     def headers(self, basic_auth=False, bearer_key=None, nonce=None):
    #         h = dict()
    #         if basic_auth:
    #             h["Authorization"] = "Basic {}".format(self.key)
    #         elif bearer_key:
    #             h["Authorization"] = "Bearer {}".format(self.secret)
    #         else:
    #             if not nonce:
    #                 from time import time

    #                 nonce = int(time())
    #             h = {
    #                 "X-Authentication-Key": self.key,
    #                 "X-Authentication-Nonce": str(nonce),
    #                 "X-Authentication-Signature": b64encode(
    #                     hmac.new(
    #                         self.secret.encode(), msg=str(nonce).encode(), digestmod=sha256
    #                     ).digest()
    #                 ),
    #             }

    #         return h

    async def process_job(
        self,
        job: Job,
        **kwargs,
    ) -> Job:
        url = job.inputs["url"]
        job.outputs = []  # this doesn't actually work.
        async with httpx.AsyncClient() as client:
            while url:
                response = await client.get(url, params=job.parameters)
                response.raise_for_status()  # Raise an exception for bad status codes

                data = response.json()
                job.outputs.append(data["data"])

                # Check for pagination information
                paging = data.get("paging")
                url = paging.get("next") if paging else None

        return job
