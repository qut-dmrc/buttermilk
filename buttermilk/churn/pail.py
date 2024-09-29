

from pydantic import BaseModel


class Pail(BaseModel):
    pass

    def _get_examples(self, record_id, identifiers, n):
        return []