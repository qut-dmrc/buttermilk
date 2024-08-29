from langchain_community.llms import Replicate


# Replicate allows us to access llama through a hosted api service
def replicatellama2(**kwargs):
    options = dict(
        model="a16z-infra/llama-2-13b-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
        input={"temperature": 1.0, "max_length": 4000, "top_p": 1},
    )
    options.update(**kwargs)
    return Replicate(**options)


def replicatellama3(**kwargs):
    options = dict(
        model="meta/meta-llama-3-70b-instruct",
        input={"temperature": 1.0, "max_length": 4000, "top_p": 1},
    )
    options.update(**kwargs)
    return Replicate(**options)