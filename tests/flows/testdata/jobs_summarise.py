job1 = dict(
    job_id="eeJNmVmUjeTHphAEr4Mi4c",
    timestamp="2024-02-02 00:03:40Z",
    source=["osb development"],
    run_info=dict(
        run_id="20241119T2158Z-AZAv-J5HW6L4KT6-suzor",
        ip="204.93.149.229",
        node_name="J5HW6L4KT6",
        username="suzor",
        save_dir="/var/folders/m0/dnlbtw0563gbgn18y50l5vjm0000gq/T/tmpdgkbpb_7",
    ),
    record=dict(record_id="HaBsyRuS4PL6jDMHAHMbBM", name=None, text=None),
    parameters={
        "template": "summarise",
        "instructions": "summarise_osb",
        "model": "gemini15pro",
    },
    inputs={"instructions": "summarise_osb"},
    agent_info={
        "name": "summarise",
        "num_runs": 2,
        "concurrency": 20,
        "save": {
            "type": "bq",
            "destination": None,
            "db_schema": "/Users/suzor/src/buttermilk/buttermilk/schemas/flow.json",
            "dataset": "dmrc-analysis.toxicity.flow",
        },
        "parameters": {
            "template": "summarise",
            "instructions": "summarise_osb",
            "model": ["gpt4o", "sonnet", "gemini15pro"],
        },
        "inputs": {},
        "outputs": {},
        "connection": {
            "name": "gemini15pro",
            "connection": "API",
            "model": "gemini-1.5-pro-002",
            "obj": "VertexNoFilter",
            "configs": {
                "model": "gemini-1.5-pro-002",
                "temperature": 1.0,
                "top_p": 0.95,
            },
        },
        "model_params": {
            "model_name": "gemini-1.5-pro-002",
            "temperature": 1.0,
            "candidate_count": 1,
            "top_p": 0.95,
            "response_mime_type": "application/json",
            "_type": "vertexai",
        },
    },
    outputs=dict(
        category=None,
        prediction=None,
        result="takedown",
        labels=[],
        confidence=None,
        severity=None,
        reasons="""
            'The Violence and Incitement Community Standard prohibits "content that constitutes a credible threat to public or personal safety," 
including "calls for high-severity violence."  The standard defines "threat" as including calls for violence even when the target isn\'t specified
but is represented by a symbol or a visual depiction of a method of violence. The video depicts stone-throwing, which qualifies as "high-severity 
violence." Although the targets are not explicitly named, they are visually depicted as the person on the balcony and the building.  The crowd\'s 
chants of "maro maro" (hit/beat) constitute a call to violence directed at those targets. Therefore, the video contains a call for high-severity 
violence. Whether this call constitutes a "credible threat" depends on the context. The volatile situation in Sambalpur, with ongoing communal 
violence and government-imposed restrictions, strongly suggests that the call to violence in the video is credible because it creates a serious 
and likely risk of inciting further violence.',
            'A minority of the board disagreed, suggesting that merely showing past incitement was not a call to violence, absent contextual 
clues. They suggested that additional context, such as news reporting or awareness raising, might allow content depicting violence to be allowed.'
        """,
        scores=None,
        metadata=None,
        case_id="2024-004-FB-UA",
        title="Communal Violence in Indian State of Odisha",
        type="standard",
        topics=["Freedom of expression", "Religion", "Violence", "Incitement"],
        standards=["Violence and Incitement Community Standard"],
        location="India",
        date="2023-08-16",
        content="""A video posted on Facebook depicted a religious procession in Sambalpur, Odisha, India, during the Hindu festival of Hanuman 
Jayanti. Participants carried saffron-colored flags, associated with Hindu nationalism, and chanted "Jai Shri Ram," a phrase that has been used to
express both religious devotion and hostility towards minority groups. The video shows a person on a balcony throwing a stone at the procession, 
after which the crowd retaliates by throwing stones toward the building while chanting "Jai Shri Ram," "bhago" (run), and "maro maro" (hit/beat). 
The video\'s caption simply read "Sambalpur." The post was viewed approximately 2,000 times and received fewer than 100 comments and reactions.  
Communal violence broke out between Hindus and Muslims during the festival in Sambalpur, resulting in property damage, injuries, and one death.  
The Odisha state government imposed a curfew and internet shutdown in response.  The post was made during this period of heightened tension and 
violence.""",
        recommendations="""'Ensure that the Violence and Incitement Community Standard explicitly allows content that condemns or raises awareness of violent 
threats, even if the content would otherwise violate the Standard.',
            'Provide clearer public information about the "spirit of the policy" allowance, which permits exceptions to the Community Standards 
when the rationale and values of the platform demand a different outcome than a strict application of the rules.'
        """,
    ),
    error={},
)


job2 = dict(
    job_id="duX8ncqec4LjHtPHkMkmJN",
    timestamp="2024-02-02 00:03:40Z",
    source=["osb development"],
    run_info=dict(
        run_id="20241119T2158Z-AZAv-J5HW6L4KT6-suzor",
        ip="204.93.149.229",
        node_name="J5HW6L4KT6",
        username="suzor",
        save_dir="/var/folders/m0/dnlbtw0563gbgn18y50l5vjm0000gq/T/tmpdgkbpb_7",
    ),
    record=dict(record_id="HaBsyRuS4PL6jDMHAHMbBM", name=None),
    parameters={
        "template": "synthesise",
        "instructions": "summarise_osb",
        "content": "record.text",
        "answers": "summarise",
        "model": "gpt4o",
    },
    inputs={
        "instructions": "summarise_osb",
        "content": "record.text",
        "answers": "summarise",
    },
    agent_info={
        "name": "synthesise",
        "num_runs": 1,
        "concurrency": 20,
        "save": {
            "type": "bq",
            "destination": None,
            "db_schema": "/Users/suzor/src/buttermilk/buttermilk/schemas/flow.json",
            "dataset": "dmrc-analysis.toxicity.flow",
        },
        "parameters": {
            "template": "synthesise",
            "instructions": "summarise_osb",
            "model": ["gpt4o", "sonnet"],
            "content": "record.text",
            "answers": "summarise",
        },
        "inputs": {},
        "outputs": {},
        "connection": {
            "name": "gpt4o",
            "model": "gpt4o_instruct_azure_2024-08-06",
            "connection": "AzureChatOpenAI",
            "obj": "AzureChatOpenAI",
            "configs": {
                "azure_endpoint": "https://platformaieast.openai.azure.com",
                "deployment_name": "platformai-gpt4o",
                "openai_api_version": "2024-08-01-preview",
                "temperature": 1.0,
            },
        },
        "model_params": {
            "azure_deployment": "platformai-gpt4o",
            "model_name": None,
            "model": None,
            "stream": False,
            "n": 1,
            "temperature": 1.0,
            "_type": "azure-openai-chat",
        },
    },
    outputs=dict(
        category=None,
        prediction=None,
        result="takedown",
        labels=[],
        confidence=None,
        severity=None,
        reasons=[
            """
            "The Violence and Incitement Community Standard prohibits content that constitutes a credible threat to public or personal safety, 
including calls for high-severity violence. The standard applies to situations where a target is represented by a symbol or a method of violence 
is depicted. The video's depiction of stone-throwing and the crowd's chants of 'maro maro' (hit/beat) fulfill these criteria of high-severity 
violence, with the individual on the balcony and the building portrayed as implicit targets.",
            "The context of the video significantly contributes to its credibility as a threat. It was posted during a period of heightened 
communal violence in Sambalpur, where tensions between Hindu and Muslim communities were already high, signified by incidents like stone-pelting. 
These processions were linked with nationalistic symbols, such as saffron flags and chants of 'Jai Shri Ram,' which have associations with 
inciting hostility against minorities in India.",
            "While some content violates the standard, exceptions exist for posts raising awareness or condemning violence. However, this video 
did not qualify for such exceptions due to its neutral caption ('Sambalpur') and lack of contextual indicators suggesting an intent to inform or 
educate. Without these elements, the content didn't clearly serve an awareness-raising or condemnatory purpose.",
            'The automated removal of all identical videos, regardless of caption, was justified due to the risk of further violence and the 
challenge of individual content review at scale amid ongoing tensions. However, these actions should be time-bound and geographically limited to 
prevent undue restrictions on freedom of expression in less affected contexts.'""",
        ],
        scores=None,
        metadata=None,
        case_id="2023-015-FB-UA",
        title="Communal Violence in Indian State of Odisha",
        type="standard",
        topics=["Freedom of expression", "Religion", "Violence", "Incitement"],
        standards=["Violence and Incitement Community Standard"],
        location="India",
        date="2023-04-13",
        content="""A Facebook video post titled 'Sambalpur' depicted a religious procession related to the Hindu festival of Hanuman Jayanti in 
Sambalpur, Odisha, India. The video showed a crowd carrying saffron flags (associated with Hindu nationalism) and chanting 'Jai Shri Ram,' a 
phrase known for expressing devotion to the Hindu god Ram and, in certain contexts, hostility against minority groups, especially Muslims. The 
video zoomed in on a person on a balcony throwing a stone at the procession, after which the crowd retaliated by throwing stones toward the 
building while chanting 'Jai Shri Ram,' 'bhago' (run), and 'maro maro' (hit/beat). The video had a caption reading 'Sambalpur'.""",
        recommendations=[
            """Ensure the Violence and Incitement Community Standard explicitly allows content that condemns or raises awareness of violent threats,
even if the content otherwise violates the Standard.',
            "Provide clearer public information about the 'spirit of the policy' allowance that permits exceptions to the Community Standards when
the rationale and values of the platform demand a different outcome than a strict application of the rules."
        """,
        ],
    ),
    error={},
    metadata={
        "outputs": {
            "token_usage": {
                "completion_tokens": 671,
                "prompt_tokens": 20643,
                "total_tokens": 21314,
                "completion_tokens_details": None,
                "prompt_tokens_details": None,
            },
            "model_name": "gpt-4o-2024-08-06",
            "system_fingerprint": "fp_04751d0b65",
            "finish_reason": "stop",
            "logprobs": None,
            "content_filter_results": {},
            "seconds_elapsed": 17.186572074890137,
        },
    },
)

job3 = dict(
    job_id="3aWSrHV4ay8VRGY3eHxxpe",
    timestamp="2024-02-02 00:03:40Z",
    source=["osb development"],
    run_info=dict(
        run_id="20241119T2158Z-AZAv-J5HW6L4KT6-suzor",
        ip="204.93.149.229",
        node_name="J5HW6L4KT6",
        username="suzor",
        save_dir="/var/folders/m0/dnlbtw0563gbgn18y50l5vjm0000gq/T/tmpdgkbpb_7",
    ),
    record=dict(record_id="HaBsyRuS4PL6jDMHAHMbBM", name=None),
    parameters={
        "template": "synthesise",
        "instructions": "summarise_osb",
        "content": "record.text",
        "answers": "summarise",
        "model": "sonnet",
    },
    inputs={
        "instructions": "summarise_osb",
        "content": "record.text",
        "answers": "summarise",
    },
    agent_info={
        "name": "synthesise",
        "num_runs": 1,
        "concurrency": 20,
        "save": {
            "type": "bq",
            "destination": None,
            "db_schema": "/Users/suzor/src/buttermilk/buttermilk/schemas/flow.json",
            "dataset": "dmrc-analysis.toxicity.flow",
        },
        "parameters": {
            "template": "synthesise",
            "instructions": "summarise_osb",
            "model": ["gpt4o", "sonnet"],
            "content": "record.text",
            "answers": "summarise",
        },
        "inputs": {},
        "outputs": {},
        "connection": {
            "name": "sonnet",
            "model": "claude-3-5-sonnet-20241022",
            "connection": "API",
            "obj": "ChatAnthropic",
            "configs": {"model": "claude-3-5-sonnet-20241022", "temperature": 1.0},
        },
        "model_params": {
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 1.0,
            "top_k": None,
            "top_p": None,
            "model_kwargs": {},
            "streaming": False,
            "max_retries": 2,
            "default_request_timeout": None,
            "_type": "anthropic-chat",
        },
    },
    outputs=dict(
        category=None,
        prediction=None,
        result="takedown",
        labels=[],
        confidence=None,
        severity=None,
        reasons=[
            """
            'The Violence and Incitement Community Standard prohibits content constituting credible threats to public safety, including calls for 
high-severity violence where no specific target is identified but a method of violence is shown. The video depicts stone throwing and calls for 
violence through chants, satisfying this definition.',
            'Several contextual factors established the credibility of the threat: (1) The content was posted during ongoing communal violence 
that had already resulted in death, (2) The religious symbols and chants shown had been associated with organized violence against Muslims, (3) 
Stone-pelting incidents during such processions have frequently triggered wider Hindu-Muslim violence, (4) Similar content was being shared in 
coordinated ways to incite violence.',
            'While the policy allows exceptions for content that condemns violence or raises awareness, this post included no contextual 
indicators of such purposes. The neutral caption "Sambalpur" provided no indication that the content was meant to inform or educate rather than 
incite.',
            "A majority found Meta's broader enforcement action of removing all identical videos through automated matching was justified given: 
(1) The immediate need to prevent violence during an ongoing crisis, (2) Evidence the original video was going viral with violating comments, (3) 
Reports of coordinated sharing of such videos to incite violence, (4) The challenges of individual review at scale during a crisis. However, such 
broad enforcement measures should be time-bound and geographically limited.",
            'A minority believed blanket removal of all identical content regardless of context was disproportionate, as it could prevent 
legitimate news reporting and awareness-raising during a crisis when information access is crucial.'""",
        ],
        scores=None,
        metadata=None,
        case_id="2023-015-FB-UA",
        title="Communal Violence in Indian State of Odisha",
        type="standard",
        topics=[
            "Violence and incitement",
            "Religious conflict",
            "Freedom of expression",
            "Content moderation at scale",
        ],
        standards=["Violence and Incitement Community Standard"],
        location="India",
        date="2023-04-13",
        content="""A Facebook user posted a video showing a religious procession in Sambalpur, India during the Hindu festival of Hanuman Jayanti. 
The video showed participants carrying saffron-colored flags (associated with Hindu nationalism) and chanting "Jai Shri Ram" (a phrase used both 
for religious devotion and, in some contexts, to express hostility toward Muslims). The footage captured someone throwing a stone from a 
building\'s balcony at the procession, followed by the crowd throwing stones back while chanting "bhago" (run) and "maro maro" (hit/beat). The 
post\'s caption simply read "Sambalpur." The content was posted during a period of communal violence that had resulted in one death, property 
damage, and a government-imposed curfew and internet shutdown.""",
        recommendations=[
            """'Meta should update the Violence and Incitement Community Standard to explicitly allow content containing statements that condemn or 
raise awareness of violent threats',
            "Meta should clarify in its Community Standards that exceptions to the rules may be made when policy rationale and Meta's values 
demand a different outcome than strict rule application""",
        ],
    ),
    error={},
    metadata={
        "outputs": {
            "id": "msg_01RbWnY4iKG5Xhkaai888UCF",
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 22209, "output_tokens": 820},
            "seconds_elapsed": 26.67640709877014,
        },
    },
)

jobs = [job1, job2, job3]

import json

print(json.dumps(jobs))
