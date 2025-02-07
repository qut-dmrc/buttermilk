from promptflow.tracing import start_trace, trace


@trace
def my_function(input: str) -> str:
    output = input + "Hello World!"
    return output


def test_trace():
    my_function("This is my function")
    start_trace(
        collection="my_custom_collection", resource_attributes=dict(job="testing")
    )


def test_azure_default():
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    # Create a secret client using the DefaultAzureCredential
    cred = DefaultAzureCredential()
    client = SecretClient(
        "https://suzorvault.vault.azure.net/",
        credential=cred,
    )

    secret = client.get_secret("testsecret").value
    assert secret
