FROM python:3.11-slim

WORKDIR /code

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=3)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
