ARG PAVEDESK_DEPS_IMAGE=python:3.11-slim
FROM ${PAVEDESK_DEPS_IMAGE} AS dependencies
USER root
WORKDIR /build/pavedesk
COPY requirements.txt .
RUN python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt && python -m pip check

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 PAVEDESK_DATA=/var/lib/pavedesk
WORKDIR /opt/pavedesk
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY pavedesk ./pavedesk
COPY static ./static
RUN useradd -U -u 11512 pavedesk && install -d -o pavedesk -g pavedesk /var/lib/pavedesk
USER pavedesk
EXPOSE 8000
CMD ["python", "-m", "pavedesk"]
