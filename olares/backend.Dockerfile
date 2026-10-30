# Capa Olares sobre la imagen upstream del backend:
#  - identidad uid/gid 1000 (los volúmenes de usuarios de Olares no son root)
#  - carpetas de datos y caché escribibles
#  - etiqueta de origen para que GHCR enlace el paquete al repo público
ARG BASE_IMAGE=aimoviestudio-backend:base
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/User-Olares/aimoviestudio" \
      org.opencontainers.image.title="AI Movie Studio 2 backend" \
      org.opencontainers.image.version="2.0.1"

ENV HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    MPLCONFIGDIR=/app/.cache/matplotlib

RUN mkdir -p /app/assets /app/core/workflows /app/.cache \
 && chown -R 1000:1000 /app

USER 1000:1000
