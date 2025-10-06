FROM python:3.9.13

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /src

ENV ACCESS_KEY= 
ENV SECRET_KEY= 

COPY . .

# List files for debugging
RUN ls -lrt

# Check if requirements file exists before installing
RUN [ -f requirements/requirements.txt ] && \
    pip3 install --no-cache-dir -r requirements/requirements.txt || \
    echo "No requirements file found, skipping pip install."


CMD ["python3", "main.py"]

