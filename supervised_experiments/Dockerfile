FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
RUN echo date

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils vim man build-essential wget sudo python3.8 python3-pip htop zlib1g-dev swig unzip libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf git-all python3.8-dev && \
    apt-get update -y && apt-get install -y texlive-full texstudio && \
    apt-get install ca-certificates -y && \
    rm -rf /var/lib/apt/lists/*

RUN python3.8 -m pip install --upgrade pip

RUN pip3.8 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip3.8 install torch_geometric

RUN pip3.8 install numpy scipy pyyaml matplotlib ruamel.yaml networkx tensorboardX gym tqdm cffi gtimer cffi Cython lockfile
RUN pip3.8 install glfw imageio
RUN pip3.8 install sacred pymongo
RUN pip3.8 install wandb 
RUN pip3.8 install scikit-learn
RUN pip3.8 install opencv-python
RUN pip3.8 install cvxpy
RUN pip3.8 install pandas
RUN pip3.8 install tueplots

RUN pip3.8 install rotograd
RUN pip3.8 install requests -U
ARG uid=1000
ARG user
RUN echo /$user $uid
RUN useradd -d /$user -u $uid $user 
USER $user

WORKDIR /$user/smto_analysis
ENV PYTHONPATH=/$user/smto_analysis

# Not ideal
ARG wandb_api_key
ENV WANDB_API_KEY=wandb_api_key

ENV PATH=/$user/.local/bin:$PATH
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV HOME=/$user/smto_analysis
ENV DATA_FOLDER=data
