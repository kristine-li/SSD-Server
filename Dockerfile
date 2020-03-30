FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -yq --no-install-recommends \
    python3 \
    python3-pip
RUN pip3 install --upgrade pip==20.0.2 \
    && pip3 install setuptools

RUN apt-get install libglib2.0-dev -y
RUN apt-get install libsm6 -y
RUN apt-get install libxrender1 -y
RUN apt-get install libxext6 -y


# for flask web server
EXPOSE 5000

# set working directory
ADD . /SSD_server
WORKDIR /SSD_server

# install required libraries

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple  -r requirements.txt

