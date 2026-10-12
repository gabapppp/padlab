FROM ubuntu:22.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build qtbase5-dev libopencv-dev ca-certificates qt5-image-formats-plugins \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends kimageformat-plugins && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY src ./src
COPY tests ./tests
COPY packaging ./packaging
RUN cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
    && cmake --build /build --parallel 2 \
    && QT_QPA_PLATFORM=offscreen ctest --test-dir /build --output-on-failure
RUN cmake --install /build --prefix /opt/padlab
RUN apt-get update && apt-get install -y --no-install-recommends file && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /release && cd /build && cpack -B /release \
    && cp /src/packaging/INSTALL.txt /opt/padlab/INSTALL.txt \
    && tar -czf /release/padlab-0.2.0-ubuntu22.04-amd64.tar.gz -C /opt/padlab . \
    && cd /release && sha256sum *.deb *.tar.gz > SHA256SUMS
FROM scratch AS artifacts
COPY --from=build /opt/padlab /
COPY --from=build /build/demo /demo
COPY --from=build /build/gui-preview.png /gui-preview.png
COPY --from=build /build/Testing/Temporary/LastTest.log /test-results.log
COPY --from=build /release /release

FROM ubuntu:22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libqt5widgets5 libqt5concurrent5 libopencv-core4.5d libopencv-imgproc4.5d \
    libopencv-imgcodecs4.5d libqt5gui5 qtwayland5 qt5-image-formats-plugins kimageformat-plugins fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/padlab /opt/padlab
ENV PATH="/opt/padlab/bin:${PATH}"
WORKDIR /data
CMD ["padlab"]
