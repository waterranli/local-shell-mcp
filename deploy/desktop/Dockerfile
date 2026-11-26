# Reproducible local runtime: upstream v5.1.1 with the existing HTTP/live-auth
# corrections, idle browser cleanup, fair result/status request delivery, and
# live credential / stale-token auth storm protection.
FROM fwerkor/local-shell-mcp:5.1.1@sha256:993b33f35db2f71e6f489112400f57ae61b9274455f77f66b067451b98c6fadb
COPY http204.patch live-channel-performance.patch browser-idle-cleanup.patch request-queue.patch live-auth-storm.patch live-reconnect-token.patch /tmp/lsm-patches/
RUN patch --batch --forward -p1 -d /app < /tmp/lsm-patches/http204.patch \
    && patch --batch --forward -p1 -d /app < /tmp/lsm-patches/live-channel-performance.patch \
    && patch --batch --forward -p1 -d /app < /tmp/lsm-patches/browser-idle-cleanup.patch \
    && patch --batch --forward -p1 -d /app < /tmp/lsm-patches/request-queue.patch \
    && patch --batch --forward -p1 -d /app < /tmp/lsm-patches/live-auth-storm.patch \
    && patch --batch --forward -p1 -d /app < /tmp/lsm-patches/live-reconnect-token.patch
COPY ui_static/live-workspace.html ui_static/live-workspace-aliases.json /app/src/local_shell_mcp/ui_static/
ARG LSM_FIX_REVISION
LABEL org.opencontainers.image.revision=$LSM_FIX_REVISION
