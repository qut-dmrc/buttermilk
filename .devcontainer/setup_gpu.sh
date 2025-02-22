export DEBIAN_FRONTEND=noninteractive

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
echo testing pytorch:
python -c "import torch;torch.cuda.current_device()"

#pip install cudf-cu12 cuml-cu12 --extra-index-url=https://pypi.nvidia.com