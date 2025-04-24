from buttermilk.agents.fetch import FetchRecord
from buttermilk.tools.characters import CharacterGenerator
from buttermilk.tools.rag.ragzot import RagZot

ANY = ["FetchRecord", "CharacterGenerator", "ragzot"]
AVAILABLE_TOOLS = {"fetch_record": FetchRecord, "character_generator": CharacterGenerator, "ragzot": RagZot}
