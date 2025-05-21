FROM python:3.12
ARG USER_NAME=dfki
ARG UID=1000
ARG GID=1000

SHELL [ "/bin/bash", "-c" ]


# Install system dependencies
RUN apt-get update && apt-get install -y libgl1

RUN true \
  && pip install --upgrade pip \
  && pip install pipx

RUN true \
  && mkdir /app \
  && groupadd -g $GID -o $USER_NAME \
  && useradd -m -u $UID -g $GID -o -s /bin/bash $USER_NAME \
  && chown $USER_NAME:$USER_NAME /app

WORKDIR /app
USER $USER_NAME

ENV PATH="${PATH}:/home/$USER_NAME/.local/bin"

RUN true \
  && pipx install poetry \
  && pipx install uvicorn \
  && pipx install gunicorn

COPY ./pyproject.toml ./poetry.toml ./poetry.lock ./

# install pytorch dependencies
RUN poetry run pip install --upgrade pip
RUN poetry run pip install torch==2.4.1 torchvision==0.19.1
# install other dependencies
RUN poetry install

COPY ./ ./

ENV GEONODE_BASE_URL=http://django:8000
EXPOSE 8000


CMD [ "bash", "-c", "poetry run app"]
