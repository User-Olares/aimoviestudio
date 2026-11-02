# Capa Olares sobre la imagen upstream del frontend (Next.js).
ARG BASE_IMAGE=aimoviestudio-frontend:base
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/User-Olares/aimoviestudio" \
      org.opencontainers.image.title="AI Movie Studio 2 frontend" \
      org.opencontainers.image.version="2.0.6"

ENV HOME=/app \
    NEXT_TELEMETRY_DISABLED=1

RUN mkdir -p /app/.cache/next \
 && chown -R 1000:1000 /app

USER 1000:1000
