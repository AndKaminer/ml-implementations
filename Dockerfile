FROM python:3.10-slim

WORKDIR /opt

COPY . .

RUN pip install poetry

RUN poetry install

WORKDIR ./ml_serving

EXPOSE 8000

ENTRYPOINT ["poetry", "run", "gunicorn", "flaskr:create_app()", "--bind", "0.0.0.0:8000"]
