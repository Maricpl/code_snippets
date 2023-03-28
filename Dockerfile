FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
RUN pip3 install pandas transformers numpy
COPY naive-baseline-task-1.py /naive-baseline-task-1.py
COPY model /model
COPY tokenizer /tokenizer
ENTRYPOINT [ "" ]