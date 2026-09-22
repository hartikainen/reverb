FROM ubuntu:24.04@sha256:786a8b558f7be160c6c8c4a54f9a57274f3b4fb1491cf65146521ae77ff1dc54

ARG TARGETARCH
ARG BAZEL_VERSION
ARG UV_VERSION=0.10.2

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git unzip zip pkg-config patchelf python3 \
    && rm -rf /var/lib/apt/lists/*
RUN case "$TARGETARCH" in \
      amd64) bazel_arch=x86_64 ;; \
      arm64) bazel_arch=arm64 ;; \
      *) exit 1 ;; \
    esac \
    && curl -fsSL "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-${bazel_arch}" -o /usr/local/bin/bazel \
    && chmod +x /usr/local/bin/bazel
RUN curl -fsSL "https://astral.sh/uv/${UV_VERSION}/install.sh" -o /tmp/install-uv.sh \
    && UV_INSTALL_DIR=/usr/local/bin sh /tmp/install-uv.sh \
    && rm /tmp/install-uv.sh
RUN useradd --create-home reverb \
    && mkdir /home/reverb/.cache \
    && chown reverb:reverb /home/reverb/.cache
COPY --chmod=755 build.sh /usr/local/bin/build-reverb-wheel
USER reverb
WORKDIR /home/reverb
ENTRYPOINT ["/usr/local/bin/build-reverb-wheel"]
