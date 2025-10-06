FROM python:3.9.13

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /src

COPY . .

# List files for debugging
RUN ls -lrt

# Check if requirements file exists before installing
RUN [ -f requirement/requirements.txt ] && \
    pip3 install --no-cache-dir -r requirements/requirement.txt || \
    echo "No requirements file found, skipping pip install."

ENV http_proxy=""
ENV https_proxy=""

CMD ["python3", "main_api.py"]
