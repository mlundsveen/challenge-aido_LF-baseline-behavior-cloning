ARG DOCKER_REGISTRY=docker.io
ARG ARCH=amd64
ARG MAJOR=daffy
ARG BASE_TAG=${MAJOR}-${ARCH}
FROM ${DOCKER_REGISTRY}/duckietown/dt-machine-learning-base-environment:${BASE_TAG}

ARG PIP_INDEX_URL="https://pypi.org/simple"
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

# Setup any additional pip packages
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN python3 -m pip install  -r .requirements.txt

# Copy all solution files to workspace
WORKDIR /submission
COPY solution.py /submission
COPY FrankNet.h5 /submission
COPY helperFncs.py /submission
COPY frankModel.py /submission

ENTRYPOINT ["python3", "solution.py"]
