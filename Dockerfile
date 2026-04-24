FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y  --no-install-recommends \
    nasm build-essential gdb && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /mnt/project
CMD ["tail", "-f", "/dev/null"]
