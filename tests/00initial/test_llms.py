from langchain.prompts import ChatPromptTemplate


def test_cheap_llm(llms, cheapchatmodel):
    llm = llms[cheapchatmodel]
    assert llm

    q = "hi! what's your name?"
    chain = ChatPromptTemplate.from_messages([("human", q)]) | llm
    answer = chain.invoke({})
    assert answer

def test_openai_key():
    import os

    import openai

    # Set your OpenAI API key
    assert os.environ['OPENAI_API_KEY'] is not None
    openai.api_key = os.environ['OPENAI_API_KEY']  # This is redundant, but let's be explicit
    openai.api_type = 'openai'

    # Define the prompt
    prompt = "Write a short story about a cat who goes on an adventure."

    # Call the OpenAI Chat API
    response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ]
    )
    text = response.choices[0].message.content
    assert text 

    # Print the generated text
    print(text)
    
    assert len(text) > 1000  # Short story should be longer than 100ish words
