import pytest

from buttermilk.llms import CHEAP_CHAT_MODELS, MULTIMODAL_MODELS

@pytest.mark.parametrize("model", CHEAP_CHAT_MODELS)
def test_frames_text(bm, example_coal, model):
    from buttermilk.flows.extract import Analyst
    flow = Analyst(template="frames.prompty", model=model)
    output = flow(content=example_coal)
    pass
    assert output

@pytest.mark.parametrize("model", CHEAP_CHAT_MODELS)
def test_framing_climate(bm, example_coal, model):
    from buttermilk.flows.extract import Analyst
    flow = Analyst(template="frames.prompty", model=model)
    output = flow(content=example_coal)
    pass
    assert output

@pytest.mark.parametrize("model", MULTIMODAL_MODELS)
def test_framing_video(model, link_to_video):
    from buttermilk.flows.video.video import Analyst
    flow = Analyst(template="frames_system.jinja2", model='gemini15pro')
    output = flow(content='see video', media_attachment_uri=link_to_video)
    pass
    assert output

@pytest.fixture(scope="session")
def link_to_video():
    return "gs://dmrc-platforms/test/fyp/tiktok-imane-01.mp4"

@pytest.fixture(scope="session")
def example_coal():
    return """Scott Morrison brings coal to question time: what fresh idiocy is this?
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

Perhaps they could just shut up and get on with it."""
