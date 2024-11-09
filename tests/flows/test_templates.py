import re
import pytest
from buttermilk.utils.templating import make_messages 

from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk.agents.lc import LC

from langchain_core.prompts import    ChatMessagePromptTemplate, ChatPromptTemplate

# answers:
# type: list
# keys:
#   id: string
#   prediction: string
#   labels: list
#   reasons: string
#   model: string
#   template: string
#   criteria: string
SAMPLE_ANSWERS = [
    {
        "id": "1",
        "prediction": "prediction 1",
        "labels": ["label 1"],
        "reasons": "reasons 1",
        "model": "model 1",
        "template": "template 1",
        "criteria": "criteria 1"
    },
    {
        "id": "2",
        "prediction": "prediction 2",
        "labels": ["label 2"],
        "reasons": "reasons 2",
        "model": "model 2",
        "template": "template 2",
        "criteria": "criteria 2"
    }
]
EXPECTED_MESSAGES = [
    ('system', 'You are a careful auditor assessing the output of a group of experts. Your analysis will be used to evaluate the reliability of each expert. It is critical that we understand where the experts differ in their findings and why.\n\nYou will be given a series of responses. Your task is to identify the differences among the reasoning given in the expert answers.\n- You should focus on differences in reasoning and understanding of the content, not the overall conclusion.\n- Where you identify a difference in opinion, you must identify the specific reasons that experts relied on to reach different conclusions about identical content.\n\nYou should be granular and precise in your analysis. For example:\n- do not say: \'Some experts found the content to be offensive\'; \n- instead, say: \'Answers #x and #y concluded that the <specific quote> in the content was hateful towards <specific group>. Most other answers concluded that the content was a bare factual report and not offensive in itself, even though it reported on offensive material. Answer #z did not consider this point.\'\n \nYou must specifically focus on:\n- differences in how the content was interpreted and understood\n- any divergent conclusions based on applying the criteria to the content\n- instances where only some experts identified a potential breach of the criteria\n\nYou must be succinct. Importantly:\n- Do not include conclusions where consensus was unanimous.\n- Do not merely list differences; you must do your best to explain in detail the basis of the disagreement. \n- Do not provide any comment on the depth or detail level of the responses. \n- Do not worry about differences in expression or style. \n- Do not comment on the expert\'s stylistic or expressive choices.\n- Do not provide an overall conclusion; only your detailed analysis of any disagreement is useful.\n\nYou should work carefully step-by-step:\n1. Read all of the answers to identify all material findings and analytical points identified across all the responses.\n2. Group similar points from different experts together.\n3. For each point where there is some disagreement:\n    - summarise the majority / minority positions and identify any outliers.\n    - explain in detail the differences in reasoning or understanding that led experts to different conclusions.\n\n<REQUIRED FORMAT>\nYou must reply only in a fully valid JSON format.\n\n\n```json\n{\n    "analysis": [\n        { "heading": <STRING: a short description of the key point>\n          "reasons": <STRING: your detailed summary of differences in reasoning across the answers on this key point>\n        }, \n        ... <other key points> ...\n    ]\n}\n```\n\nEnsure the response follows JSON formatting rules precisely, as it will be validated and type-checked for compliance. Make sure you escape any double quotes in your response. Do not nest reasons; each heading should only relate to one step of reasoning in a sequence. Do not include any text outside of the JSON object in your response -- additional text will invalidate the answer.\n<END REQUIRED FORMAT>'),
    ("user",'Analyse the underlying reasons behind differences among the answers below.\n\n<BEGIN ORIGINAL INSTRUCTIONS>\nAlways use a transgender person\'s chosen name. Unless a former name is newsworthy or pertinent, use the name and pronouns preferred by the transgender person. If no preference is known, use the pronouns consistent with the way the subject lives publicly.\nDo not identify someone as transgender unless it provides necessary and relevant clarity or context.\n  - Do not out transgender people in your reporting.\n  - Avoid putting transgender people or their families at risk by providing unecessary identifying details, even if they live openly as trans in their community.\nDo not use offensive terms to refer to trans people.\nAvoid language that perpetuates negative or incorrect perceptions of gender identities.\nAvoid language that casts doubt on or diminishes a person\'s gender identity.\n  - For example, avoid phrasing that diminishes gender identity or sexuality with terms like "identifies as"; comparisons to "real" or "biological" men or women; references to "lifestyle" or "sexual preference".\n  - When writing about a transgender person\'s chosen name, do not say "she wants to be called," "she calls herself," "she goes by Marisol," or other phrases that cast doubt on a transgender person\'s gender.\n  - Do not put quotation marks around either a transgender person\'s chosen name or the pronoun that reflects that person\'s gender identity.\nIt is acceptable to include an explanation of lesser-known terminology, but avoid explanations of common terms that treat transgender people as abnormal or other. Some people use less common terms to describe body parts or biological functions (chestfeeding as an alternative to breastfeeding or nursing, birthing people instead of mothers or parents, etc.). These terms are not endorsed by trans people universally; use them only in direct quotes from trans people who do.\nDo not publish false narratives about trans people. For example, do not uncritically repeat claims about "grooming", "rapid-onset gender syphoria", or other debunked and harmful myths.\n  - Where it is necessary to report on accusations or disparaging comments, be clear about the evidence provided and explain any mistaken assumptions.\nAvoid cliches. Comparisons to "losing a loved one" are inappropriate if the trans person in question has not died.\nAvoid focusing predominantly on medical transition. Transgender people’s medical choices and personal histories are frequently objects of salacious scrutiny, and they should not be brought up by journalists unless directly relevant to a story. Put the person at the center of your story, in the context of family, friends and daily life.\nBe aware of how the systemic patterns of prejudice and discrimination against transgender people can inform the context of your story. Because prejudice and discrimination are so common, transgender people are far more likely than cisgender (i.e., non-transgender) people to be living in poverty, and too many experience homelessness at some point in their lives. This can make them especially vulnerable both to violence and to contact with law enforcement, with or without cause.\n<END ORIGINAL INSTRUCTIONS>\n\n\n    <BEGIN DRAFT ANSWER 1>\n    Answer: prediction 1    Labels: [\'label 1\']    Reasons:\n    reasons 1\n    <END DRAFT ANSWER 1>\n    <QA TRACKING CODE: IGNORE>\n        1 model 1 template 1 criteria 1\n    <END QA TRACKING CODE>\n    <BEGIN DRAFT ANSWER 2>\n    Answer: prediction 2    Labels: [\'label 2\']    Reasons:\n    reasons 2\n    <END DRAFT ANSWER 2>\n    <QA TRACKING CODE: IGNORE>\n        2 model 2 template 2 criteria 2\n    <END QA TRACKING CODE>')]
    
@pytest.mark.integration
def test_template_interactive():
    record = RecordInfo(content="test content", answers=SAMPLE_ANSWERS, meta={"One":"two"})
    job = Job(record=record, source='testing', parameters={'answers': 'answers', 'meta': 'meta'})
    flow = LC(flow="tes", **{'name': 'lc', 'template': 'differences', 'criteria': 'trans_simplified', 'model': 'haiku'})

    messages = flow.prepare_messages(job)
    for i, (user, msg) in enumerate(messages):
        assert user == EXPECTED_MESSAGES[i][0]
        assert re.sub(r'\W', '', msg) == re.sub(r'\W', '', EXPECTED_MESSAGES[i][1])
    # assert all([x==y for x,y in zip(messages, EXPECTED_MESSAGES)])

@pytest.mark.integration
def test_template_converted_interactive():
    record = RecordInfo(content="test content", answers=SAMPLE_ANSWERS, meta={"One":"two"})
    job = Job(record=record, source='testing', parameters={'answers': 'answers', 'meta': 'meta'})
    flow = LC(flow="tes", **{'name': 'lc', 'template': 'differences', 'criteria': 'trans_simplified', 'model': 'haiku'})

    messages = flow.prepare_messages(job)
    tpl = ChatPromptTemplate.from_messages(
                messages, template_format="jinja2"
            )
    breakpoint()
    pass

def test_make_messages_with_valid_input():
    # Test with valid input
    input_data = {
        'key1': 'value1',
        'key2': 'value2'
    }
    expected_output = 'Expected template output'  # Replace with the actual expected output
    result = make_messages(input_data)
    assert result == expected_output

def test_make_messages_with_empty_input():
    # Test with empty input
    input_data = {}
    expected_output = 'Expected template output for empty input'  # Replace with the actual expected output
    result = make_messages(input_data)
    assert result == expected_output

def test_make_messages_with_missing_keys():
    # Test with missing keys in input
    input_data = {
        'key1': 'value1'
    }
    expected_output = 'Expected template output for missing keys'  # Replace with the actual expected output
    result = make_messages(input_data)
    assert result == expected_output

def test_make_messages_with_invalid_input():
    # Test with invalid input
    input_data = 'invalid_input'
    with pytest.raises(TypeError):
        make_messages(input_data)

def test_make_messages_with_special_characters():
    # Test with special characters in input
    input_data = {
        'key1': 'value1!@#$%^&*()',
        'key2': 'value2<>?:"{}|'
    }
    expected_output = 'Expected template output for special characters'  # Replace with the actual expected output
    result = make_messages(input_data)
    assert result == expected_output
