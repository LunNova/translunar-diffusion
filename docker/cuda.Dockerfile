FROM mambaorg/micromamba:focal

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
USER root
RUN apt update && apt dist-upgrade -y && apt install libsqlite3-0 git -y && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 build-essential mpich libmpich-dev libaio-dev python3-mpi4py wget nano vim
COPY install-cuda.sh /root/
RUN /root/install-cuda.sh
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]
USER $MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER env-cuda.yml /home/micromamba/env.yml
RUN micromamba install -y -n base -f /home/micromamba/env.yml
WORKDIR /content/
COPY --chown=$MAMBA_USER:$MAMBA_USER preload.py /home/micromamba/preload.py
RUN bash -c ". /usr/local/bin/_activate_current_env.sh && id && cd /content && python /home/micromamba/preload.py"
