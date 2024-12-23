# buttermilk: opinionated data tools for HASS scholars

**AI and data tools for HASS researchers, putting culture first.**

Developed by and for @QUT-DMRC scholars, this repo aims to provide standard **flows** that help scholars make their own **pipelines** to collect data and use machine learning, generative AI, and computational techniques as part of rich analysis and experimentation that is driven by theory and deep understanding of cultural context. We try to:

* Provide a set of research-backed analysis tools that help scholars bring cultural expertise to computational methods.
* Help HASS scholars with easy, well-documented, and proven tools for data collection and analysis.
* Make ScholarOps™ easier with opinionated defaults that take care of logging and archiving in standard formats.
* Create a space for collaboration, experimentation, and evaluation of computational methods for HASS scholars.

```md
Q: Why 'buttermilk'??
A: It's cultured and flows...
```

```md
Q: What's MLOps?
A: A general term for standardised approaches to machine learning workflows that 
helps you organize your project, collaborate, iteratively improve your analysis 
and track versioned changes, monitor onging performance, reproduce experiments, 
and verify and compare results. 
```

The "pipeline" we are building is documented and versioned. We're aiming to make it easy for HASS scholars to use AI tools in a way that is understandable, traceable, and reproducible.

## Usage

So far, we have a few standard pieces that set sensible defaults to make it easy for HASS scholars to use, store, assess, compare, and reproduce complicated AI workflows in their research.

Working right now:

* Multimodal support for current-generation foundation models (Gemini, Claude, Llama, GPT) and plug-in support for basically any other analysis tool API.
* A prompt templating system that allows you to evaluate, improve, and reuse components. These will ideally become a collection of versioned prompts that have been carefully evaluated, and come with test cases to monitor if they start to break.
* Standard cloud logging, flexible data storage, secure project and individual credentials (Azure KeyVault or Google Secrets), built-in database storage (BigQuery), tracing (Promptflow or Langchain).
* An API and CLI that integrates each modular component and orchestrates complex workflows, including parallel runs and multi-step workflows.
* Use the same code to run locally, on a remote GPU, or in Azure/Google Compute [and AWS llambda, I guess, but not yet].

Next:

* Some tutorial workbooks showing a full walkthrough of an entire pipeline research run.
* A pub/sub distributed queue system and an interface to add batch runs
* A web interface and notebooks with good examples of how to assess, track, and compare performance on individual examples and aggregated runs.

## Very early stages!

We would love your help! Contact [nic](mailto:n.suzor@qut.edu.au) to discuss what you'd like to see, help us plan, or how to contribute!

## Installation

Create a new environment and install using poetry:

```shell
conda create --name bm -y -c conda-forge -c defaults python==3.11 poetry ipykernel google-crc32c

conda activate bm
poetry install --with dev
```

Authenticate to cloud providers, where all your secrets are stored.

```shell
gcloud auth login --update-adc --force
gcloud config set project [projectid]
gcloud auth application-default set-quota-project [project]
```

Configurations are stored as YAML files in `conf/`. You can select options at runtime using [hydra](https://hydra.cc).

### Dependencies and example for GPU prediction (on Ubuntu 22.04)

```shell
#!/bin/sh
export USER=ubuntu
export DEBIAN_FRONTEND=noninteractive
export INSTALL_DIR=/mnt  # change as appropriate

apt update && apt -y --no-install-recommends install neovim git zsh tmux curl pigz gnupg less nmap openssh-server python3 python3-pip rsync htop build-essential gcc g++ make psmisc keychain bmon jnettop ca-certificates ncdu software-properties-common

# nvidia toolkit and nvidia driver
# Use an image with this preinstalled, lots easier.
#wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update && sudo apt-get -y install cuda-toolkit-12-4
#sudo apt install -y cuda-drivers


# install gcloud sdk
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y

# azure cli
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# miniconda
mkdir -p $INSTALL_DIR/miniconda3 && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $INSTALL_DIR/miniconda3/miniconda.sh && bash $INSTALL_DIR/miniconda3/miniconda.sh -b -u -p $INSTALL_DIR/miniconda3 && rm $INSTALL_DIR/miniconda3/miniconda.sh && $INSTALL_DIR/miniconda3/bin/conda init


# Change cache location (our GPU machine has limited space on /)
echo -e "export XDG_CACHE_HOME=$INSTALL_DIR/cache\nexport POETRY_CACHE_DIR=$INSTALL_DIR/cache/poetry"|tee -a /home/$USER/.bashrc
echo -e "envs_dirs:\n  - $INSTALL_DIR/miniconda3/envs\npkgs_dirs:\n  - $INSTALL_DIR/miniconda3/pkgs" | tee /home/$USER/.condarc

mkdir -p $INSTALL_DIR/src $INSTALL_DIR/cache && chown -R $USER $INSTALL_DIR/cache $INSTALL_DIR/miniconda3 $INSTALL_DIR/src

# create environment
$INSTALL_DIR/miniconda3/bin/conda create --name bm -y -c conda-forge -c pytorch -c nvidia python==3.11 poetry ipykernel google-crc32c pytorch torchvision torchaudio pytorch-cuda=12.4 && chown -R ubuntu $INSTALL_DIR/miniconda3/envs
```

At this point, log out and back in to activate the environment and check your install:

```shell
conda activate bm
nvidia-smi

export INSTALL_DIR=/mnt  # change as appropriate
# checkout repository
mkdir -p $INSTALL_DIR/src && cd $INSTALL_DIR/src && git clone https://github.com/qut-dmrc/buttermilk.git

# install dependencies
cd buttermilk
POETRY_CACHE_DIR=/mnt/cache/poetry poetry install --with dev
# set up cloud connections
az login
gcloud auth login --enable-gdrive-access --update-adc --force
pf config set connection.provider=azureml://subscriptions/7e7e056a-4224-4e26-99d2-1e3f9a688c50/resourcegroups/rg-suzor_ai/providers/Microsoft.MachineLearningServices/workspaces/automod
pf config set trace.destination=azureml://subscriptions/7e7e056a-4224-4e26-99d2-1e3f9a688c50/resourcegroups/rg-suzor_ai/providers/Microsoft.MachineLearningServices/workspaces/automod

# probably need to set some environment variables
echo -e "HF_HUB_ENABLE_HF_TRANSFER=1\nPOETRY_CACHE_DIR=/mnt/cache/poetry\nHF_HOME=/mnt/cache/hf\nPF_WORKER_COUNT=24\nPF_BATCH_METHOD=fork" | tee -a /mnt/src/buttermilk/.env
```

### Run

Serially, on a gpu:
`python -m examples.automod.pfmod +experiments=ots_gpu +data=drag_noalt +save=bq`

Parallel:
`python -m examples.automod.pfmod --multirun hydra/launcher=joblib +experiments=ots +data=drag_noalt +save=bq experiments.moderate.init.flow=GoogleModerate,NemoInputSimpleGPT4o,NemoInputComplexGPT4o,NemoOutputSimpleGPT4o,NemoOutputComplexGPT4o,NemoInputSimpleLlama31_70b,NemoInputComplexLlama31_70b,NemoOutputSimpleLlama31_70b,NemoOutputComplexLlama31_70b,Comprehend,Perspective,AzureContentSafety,OpenAIModerator,LlamaGuard1Replicate,LlamaGuard1Together,LlamaGuard2Replicate,LlamaGuard2Together,LlamaGuard3Together`