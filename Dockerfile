FROM ubuntu:16.04
RUN apt-get update
RUN apt-get -y install software-properties-common python-software-properties
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install --fix-missing -y wget curl unzip python3.6 
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6

RUN apt purge -y python2.7-minimal
RUN ln -s /usr/bin/python3.6 /usr/bin/python

RUN apt-get install libgomp1 

RUN pip install --upgrade pip
RUN pip install setuptools
RUN pip install numpy==1.17.4 argparse
RUN pip install scipy scikit-learn scikit-optimize xgboost pandas  matplotlib datetime
RUN pip install pygithub
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*


COPY src /opt/program


RUN python --version
RUN $(head -1 `which pip` | tail -c +3) --version

