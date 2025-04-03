import hydra
import pytest
from pytest import MarkDecorator

from buttermilk._core.runner_types import Record
from buttermilk.bm import BM
from buttermilk._core.llms import CHATMODELS, CHEAP_CHAT_MODELS, MULTIMODAL_MODELS, LLMs
from buttermilk.utils.media import download_and_convert
from buttermilk.utils.utils import read_file


@pytest.fixture(scope="session", autouse=True)
def objs():
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="testing")
    # Hydra will automatically instantiate the objects
    objs = hydra.utils.instantiate(cfg)
    return objs


@pytest.fixture(scope="session", autouse=True)
def bm(objs):
    return objs.bm


@pytest.fixture(scope="session", autouse=True)
def flow(objs):
    return objs.flows["test"]


@pytest.fixture(scope="session")
def logger(bm):
    return bm.logger


@pytest.fixture(scope="session")
def llms(bm: BM) -> LLMs:
    return bm.llms


def model_name(request) -> str:
    return request.param


@pytest.fixture(params=MULTIMODAL_MODELS)
def llm_multimodal(request, bm: BM):
    return bm.llms[request.param]


@pytest.fixture(params=CHEAP_CHAT_MODELS)
def llm(request, bm: BM):
    return bm.llms[request.param]


@pytest.fixture(params=CHATMODELS)
def llm_expensive(request, bm: BM):
    return bm.llms[request.param]


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def image_bytes() -> bytes:
    return read_file("tests/data/Rijksmuseum_(25621972346).jpg")


@pytest.fixture(scope="session")
def video_bytes(video_url: str) -> bytes:
    return read_file(video_url)


MEDIA_RECORDS = [
    ("sad robot local image", "tests/data/sadrobot.jpg", "image/jpeg"),
    ("web image", "https://picsum.photos/64", "image/jpeg"),
    (
        "web video",
        "https://github.com/chthomos/video-media-samples/raw/refs/heads/master/big-buck-bunny-480p-30sec.mp4",
        "video/mpeg4",
    ),
    ("gcs video", "gs://dmrc-platforms/data/tonepolice/v2IF1Kw4.mp4", "video/mpeg4"),
    ("rijksmuseum local", "tests/data/Rijksmuseum_(25621972346).jpg", "image/jpeg"),
    (
        "fight no more forever",
        """Tell General Howard I know his heart. What he told me before, I have it in my heart. I am tired of fighting. Our Chiefs are killed; Looking Glass is dead, Ta Hool Hool Shute is dead. The old men are all dead. It is the young men who say yes or no. He who led on the young men is dead. It is cold, and we have no blankets; the little children are freezing to death. My people, some of them, have run away to the hills, and have no blankets, no food. No one knows where they are - perhaps freezing to death. I want to have time to look for my children, and see how many of them I can find. Maybe I shall find them among the dead. Hear me, my Chiefs! I am tired; my heart is sick and sad. From where the sun now stands I will fight no more forever.""",
        "text/plain",
    ),
]

TEXT_RECORDS = [
    (
        "osb blm",
        """An image depicting a caricature of a Jewish man with an exaggerated hooked nose and a Star of David marked with "Jude" (resembling Holocaust-era badges), holding a music box labeled "media." A monkey labeled "BLM" sits on the man's shoulder.""",
        "text/plain",
    ),
    (
        "lady macbeth",
        """The raven himself is hoarse
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
To cry “Hold, hold!”""",
        "text/plain",
    ),
]

NEWS_RECORDS = [
    (
        "web page",
        "https://www.abc.net.au/news/2025-01-16/jewish-palestinian-australia-gaza/104825486",
        "text/html",
    ),
    (
        "coal",
        """Scott Morrison brings coal to question time: what fresh idiocy is this?
This article is more than 7 years oldKatharine Murphy
Katharine Murphy
What a bunch of clowns, hamming it up – while out in the real world an ominous and oppressive heat just won’t let up

There is no way you can write the sentence, “The treasurer of Australia, Scott Morrison, came to question time with a lump of coal on Thursday,” and have that sentence seem anything other than the ravings of a psychedelic trip, so let’s just say it and be done with it.

Scott Morrison brought coal into the House of Representatives. A nice big hunk of black coal, kindly supplied by the Minerals Council of Australia.

“This is coal,” the treasurer said triumphantly, brandishing the trophy as if he’d just stumbled across an exotic species previously thought to be extinct.

“Don’t be afraid,” he said, soothingly, “don’t be scared.”

No one was afraid, or scared. People were just confused. What was this fresh idiocy?

Barnaby Joyce juggles coal as Labor and Coalition clash over blackouts – as it happened
Read more
Journalists sitting above the fray blinked, as if this might have been a mirage. I’m still not entirely convinced it wasn’t: a cabinet minister – the treasurer, no less – with a lump of coal, in the people’s house in 2017.

The coal was passed from hand to hand, or some hands, anyway. Some pointedly declined to touch the lump, which, to be frank, didn’t look that appealing.

The coal was produced as a totem of how the government in Canberra was going to keep the lights on, and keep power prices low, and stop the relentless march of socialism, or prevent random thought crimes against base-load power stations.

Whatever the question was in the parliament on Thursday, the answer was: “South Australia has just had a blackout and, if Bill Shorten becomes the prime minister, all the lights will go off around the country.”

Australia would become like South Australia, “struggling all too often in the dark” as the prime minister said at one point – because Labor was drunk on renewable energy, or suffering from coal-o-phobia, the fear of black rocks.

It really was quite the jolt. In the space of 24 hours, question time had gone from hosting a bout of epic prime ministerial fury to pure bathos.

A bit of breaking news for the chortling Liberal students’ club in question time: energy policy is serious – how we keep the lights on, how we transform the electricity grid to function in the low-emissions future Australia has signed up to courtesy of the Paris agreement, how we rise to the great technological challenges of our age.

These are big thoughts, big questions, which on Thursday were reduced to a parody, a bit of smug culture war in the parliament – which also undercuts the government’s major political objective this week, which was to try to persuade voters that they actually live on the planet and care about the things voters care about.

How Malcolm Turnbull could ignore the facts and fund the myth of 'clean' coal
Read more
What a bunch of clowns, hamming it up, when real people are suffering blackouts in South Australia, when the country is sweltering through a summer that feels ominous and oppressive – that heat that just won’t let up, almost as if those pesky climate scientists might be on to something.

Morrison, with his big grin and his big lump of black coal – as if that meant anything, solved anything, contributed anything, to anyone or anything.

More breaking news: no one cares if coal is now the social glue of the Turnbull government, the water-cooler small talk bringing a fractious government together.

Out in the real world, no one cares that coal, for red-meat conservatives, is now not only a just means to an end in a carbon-intensive economy, the necessary stuff of economic development and jobs, but a symbolic strike against progressivism, a big middle digit to people who drive a Prius.

No one cares.

Voters do care if a government is working up a coherent policy to address the challenges we face in executing a profound economic transformation, an energy transformation that is critical to Australia’s future.

They care if policy looks after the national interest, if it makes sense, if it allows for an orderly transition from carbon-intensive energy sources to low-emissions energy sources in a way that disruption is kept to a minimum.

That, they care about.

And if we move beyond the stunts to the substance then, so far, this government is all over the shop.

We are a signatory to the Paris climate agreement but we are also flirting with subsidising new “clean” coal power stations, which, frankly, aren’t that clean.

South Australian energy minister: ‘Clean coal is a fairytale’ Guardian
The government is demanding that the states allow more gas development while ruling out a policy mechanism recommended by all its experts – an emissions intensity scheme – that would actually favour gas in the energy mix.

The government is declaring piously that we need to drive ideology out of the energy debate, and have a technology-neutral energy policy starting point, while pursuing an ideological campaign against renewables, which we will need to deliver our Paris target – the one this government signed us up to.

The official line from the government on energy policy is when it comes to technologies, it should be “all of the above” – which would be laudable, terrific even, if we actually had any coherent policy mechanism to deliver all of the above.

Cabinet’s energy subcommittee met in Canberra for the first time on Thursday afternoon. They’ve got a job of work to do.

Perhaps they could do us all a favour.

Perhaps they could just shut up and get on with it.""",
        "text/plain",
    ),
]


@pytest.fixture(
    scope="session",
    params=MEDIA_RECORDS,
    ids=[x[0] for x in MEDIA_RECORDS],
)
async def multimodal_record(request) -> Record:
    record = await download_and_convert(
        request.param[1],
        mime=request.param[2],
        title=request.param[0],
    )
    return record


@pytest.fixture(
    scope="session",
    params=NEWS_RECORDS,
    ids=[x[0] for x in NEWS_RECORDS],
)
async def news_record(request) -> Record:
    record = await download_and_convert(
        request.param[1],
        mime=request.param[2],
        title=request.param[0],
    )
    return record


@pytest.fixture
def fight_no_more_forever() -> Record:
    return Record(
        data="""Tell General Howard I know his heart. What he told me before, I have it in my heart. I am tired of fighting. Our Chiefs are killed; Looking Glass is dead, Ta Hool Hool Shute is dead. The old men are all dead. It is the young men who say yes or no. He who led on the young men is dead. It is cold, and we have no blankets; the little children are freezing to death. My people, some of them, have run away to the hills, and have no blankets, no food. No one knows where they are - perhaps freezing to death. I want to have time to look for my children, and see how many of them I can find. Maybe I shall find them among the dead. Hear me, my Chiefs! I am tired; my heart is sick and sad. From where the sun now stands I will fight no more forever.""",
        mime="text/plain",
        title="fight no more forever",
    )


@pytest.fixture(
    scope="session",
    params=TEXT_RECORDS,
    ids=[x[0] for x in TEXT_RECORDS],
)
async def text_record(request) -> Record:
    record = await download_and_convert(
        request.param[1],
        mime=request.param[2],
        title=request.param[0],
    )
    return record


VIDEO_URIS = [
    (
        "web",
        "https://github.com/chthomos/video-media-samples/raw/refs/heads/master/big-buck-bunny-480p-30sec.mp4",
    ),
]


@pytest.fixture(
    scope="session",
    params=[x[1] for x in VIDEO_URIS],
    ids=[x[0] for x in VIDEO_URIS],
)
def video_url(request) -> str:
    return request.param


@pytest.fixture
def job_minimal(fight_no_more_forever):
    """Provide a minimal job-like structure for tests."""
    return {
        "source": "testing",
        "flow_id": "testflow",
        "record": fight_no_more_forever,
        "run_info": {"info": "test_run_info"},
    }


def skipif_no_gpu(reason: str = "No GPU available") -> MarkDecorator:
    """Convenience for pytest.mark.skipif() in case no GPU is available.

    :param reason: The reason for skipping the test.
    :return: A Pytest skipif mark decorator.
    """
    from torch import cuda

    has_gpu = cuda.is_available() and cuda.device_count() > 0
    return pytest.mark.skipif(not has_gpu, reason=reason)


def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="run gpu and memory intensive tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpu"):
        # --gpu given in cli: do not skip gpu and memory intensive tests
        return
    skipgpu = pytest.mark.skip(reason="need --gpu option to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skipgpu)
