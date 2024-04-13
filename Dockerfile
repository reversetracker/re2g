FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV AZURE_STORAGE_CONNECTION_STRING="YOUR-MOTHER"

COPY requirements.txt /opt/re2g/requirements.txt
RUN pip install -r /opt/re2g/requirements.txt

COPY . /opt/re2g
WORKDIR /opt/re2g

CMD ["python", "train.py"]
