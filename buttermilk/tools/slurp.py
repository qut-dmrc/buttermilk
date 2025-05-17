######
# Slurp all data we can from an api.
#
# Input:
#
#   data: a dictionary, json string, list of dicts, or dataframe with already collected data
#       OR
#   uri: the api endpoint
#   parameters: a dict of parameters

#
# Output:
#   Data saved neatly in a standard GCS location
#   Full logging informatoin saved to Google Cloud Logging


import httpx

from buttermilk._core.agent import SingleAgent  # Import SingleAgent
from buttermilk._core.contract import AgentInput, AgentTrace  # Import AgentInput and AgentTrace


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

    async def process_job(  # Consider renaming this method to 'process' to align with Agent base class
        self,
        message: AgentInput,  # Changed parameter name and type
        **kwargs,
    ) -> AgentTrace:  # Changed return type
        url = message.inputs["url"]  # Access inputs from message
        outputs_list = []  # Use a local list to collect outputs
        async with httpx.AsyncClient() as client:
            while url:
                response = await client.get(url, parameters=message.parameters)  # Access parameters from message
                response.raise_for_status()  # Raise an exception for bad status codes

                data = response.json()
                outputs_list.append(data.get("data"))  # Append data to the list, handle missing key

                # Check for pagination information
                paging = data.get("paging")
                url = paging.get("next") if paging else None

        # Create an AgentTrace object to return the results
        trace = AgentTrace(
            agent_id=self.agent_id,
            session_id=self.session_id,  # session_id is required for AgentTrace
            agent_info=self._cfg,  # agent_info is required for AgentTrace
            inputs=message,  # Include the original input message
            outputs=outputs_list,  # Store the collected outputs
            # Add other relevant metadata if needed
        )

        return trace  # Return the AgentTrace object
