FROM python:3.10

ENV PYTHONPATH=/srv \
    # Keeps Python from generating .pyc files in the container
    PYTHONDONTWRITEBYTECODE=1 \
    # Turns off buffering for easier container logging
    PYTHONUNBUFFERED=1

WORKDIR /srv/

RUN \
    apt-get update && python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install --no-cache streamlit==1.13.0 altair==4.2.0

CMD ["streamlit", "run", "src/app.py"]