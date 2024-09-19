
FROM infyartifactory.jfrog.io/responsibleai-mms-kubeflow/privacy-compatible-image:v1.0
 
RUN useradd myLowPrivilegeUser
USER myLowPrivilegeUser
 
ENV DEBIAN_FRONTEND=noninteractive
 
ENV http_proxy=http://proxy.threatpulse.net:8080
ENV https_proxy=http://proxy.threatpulse.net:8080
 
Copy . .
 
RUN pip3 install -r requirements/requirement.txt -i https://infyartifactory.jfrog.io/artifactory/api/pypi/pypi-remote/simple --user --trusted-host infyartifactory.jfrog.io
 
WORKDIR src
 
CMD ["sh", "-c", "python3 main.py"]
 
 

