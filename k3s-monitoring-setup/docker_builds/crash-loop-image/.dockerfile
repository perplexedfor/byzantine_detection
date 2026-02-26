FROM alpine:latest

RUN apk add --no-cache stress-ng

CMD ["/bin/sh"]