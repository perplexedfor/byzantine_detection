FROM alpine:latest

RUN apk add --no-cache \
    netcat-openbsd \
    curl \
    coreutils

CMD ["/bin/sh"]