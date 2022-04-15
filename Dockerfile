##基础镜像
FROM  ccr.ccs.tencentyun.com/public_images/cuda10.0-cudnn7.4.2-dev-ubuntu16.04-opencv4.1.1-ev-base:latest

COPY . /usr/local/ev_sdk


RUN cd /usr/local/ev_sdk \
	&& cd /usr/local/ev_sdk && rm -rf build \
	&& mkdir -p build && cd build \
	&& cmake -DCMAKE_BUILD_TYPE=release .. && make -j4 && make install \
	&& cd /usr/local/ev_sdk/test && rm -rf build && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=release .. && make -j4 && make install
	
ENV LD_LIBRARY_PATH=/usr/local/ev_sdk/lib/:$LD_LIBRARY_PATH




