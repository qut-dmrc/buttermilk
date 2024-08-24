from azure.core.exceptions import ClientAuthenticationError
from azure.keyvault.secrets import SecretClient
from datatools.azcloud import auth

def test_azure():
    # Create a secret client using the DefaultAzureCredential
    cred = auth()
    client = SecretClient(
        "https://suzorvault.vault.azure.net/",
        credential=cred,
    )
    try:
        secret = client.get_secret("testsecret")
        print(secret)
    except ClientAuthenticationError as ex:
        print(ex.message)
