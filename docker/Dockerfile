# freezing to the correct cuda installation
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ADD .. /root/cr3dt

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get clean && apt-get update &&\
    apt-get install -y curl git libgl1-mesa-glx libglib2.0-0 

# CONDA
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja && \
    /opt/conda/bin/conda clean -ya


# CONDA ENVIRONMENT
RUN /opt/conda/bin/conda env create -f /root/cr3dt/cr3dt_docker.yaml

RUN /opt/conda/bin/conda init bash 

RUN . ~/.bashrc && conda activate cr3dt_docker && pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
