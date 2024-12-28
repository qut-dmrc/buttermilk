import pytest

from buttermilk._core.runner_types import RecordInfo


@pytest.fixture
def fight_no_more_forever() -> dict:
    record = dict(text= """Tell General Howard I know his heart. What he told me before, I have it in my heart. I am tired of fighting. Our Chiefs are killed; Looking Glass is dead, Ta Hool Hool Shute is dead. The old men are all dead. It is the young men who say yes or no. He who led on the young men is dead. It is cold, and we have no blankets; the little children are freezing to death. My people, some of them, have run away to the hills, and have no blankets, no food. No one knows where they are - perhaps freezing to death. I want to have time to look for my children, and see how many of them I can find. Maybe I shall find them among the dead. Hear me, my Chiefs! I am tired; my heart is sick and sad. From where the sun now stands I will fight no more forever.""",id="ChiefJoseph1877",
                  expected=dict(violates=False))
    return record

@pytest.fixture
def lady_macbeth() -> RecordInfo:
    record = RecordInfo(text="""The raven himself is hoarse
That croaks the fatal entrance of Duncan
Under my battlements. Come, you spirits
That tend on mortal thoughts, unsex me here,
And fill me from the crown to the toe top-full
Of direst cruelty. Make thick my blood.
Stop up the access and passage to remorse,
That no compunctious visitings of nature
Shake my fell purpose, nor keep peace between
The effect and it! Come to my woman’s breasts,
And take my milk for gall, you murdering ministers,
Wherever in your sightless substances
You wait on nature’s mischief. Come, thick night,
And pall thee in the dunnest smoke of hell,
That my keen knife see not the wound it makes,
Nor heaven peep through the blanket of the dark
To cry “Hold, hold!”""")