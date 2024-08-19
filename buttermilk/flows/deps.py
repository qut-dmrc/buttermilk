
from tempfile import NamedTemporaryFile
from promptflow.core import tool
from promptflow.client import PFClient as LocalPFClient
from promptflow.tracing import start_trace, trace

@tool
def my_python_tool(input1: str) -> str:
    return input1 + "!"

def run() -> None:
    data = """{"id": 1, "text": "text1"}
    {"id": 2, "text": "text2"}"""
    with NamedTemporaryFile(delete=False, suffix='.jsonl', mode='w') as f:
        f.write(data)
    dataset = f.name

    localclient = LocalPFClient()

    start_trace()

    columns = {'input1': r'${data.text}'}
    flow = my_python_tool
    run = localclient.run(flow=flow, data=dataset, column_mapping=columns, stream=True)
    print(run.status)

if __name__ == '__main__':
    run()