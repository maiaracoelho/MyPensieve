FROM nvcr.io/nvidia/tensorflow:21.03-tf2-py3

WORKDIR /workspace/app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "src/train.py"]
