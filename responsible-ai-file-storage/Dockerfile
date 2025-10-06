FROM python:3.9.13

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /src

COPY . .

RUN pip3 install --no-cache-dir -r requirements/requirements.txt

CMD ["python3", "main.py"]

