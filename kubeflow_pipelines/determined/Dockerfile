FROM python:3.7-slim

RUN apt-get update && apt-get install -y gcc python3-dev

RUN pip install determined-cli==0.18.1 determined==0.18.1

RUN mkdir /cmd/
WORKDIR /cmd/

COPY . /cmd/

ENTRYPOINT ["/bin/sh"]
