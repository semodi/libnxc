FROM rikorose/gcc-cmake:gcc-10

WORKDIR /lib

RUN curl https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.7.1%2Bcpu.zip -o /lib/libtorch.zip \
&& unzip libtorch.zip \
&& rm libtorch.zip

RUN wget https://github.com/google/googletest/archive/release-1.10.0.zip  \
&& unzip release*.zip  \
&& mv googletest*/ googletest/ \
&& rm release*.zip

RUN cd /lib/googletest \
&& cmake . \
&& make

COPY src/ /lib/src/
RUN  mkdir build
COPY arch.make.docker /lib/arch.make
COPY utils/Makefile /lib/build/
RUN cd build && make -j

COPY test/ /lib/test/
COPY models/ /lib/models/
RUN cd /lib/test/gtest/ && make docker

WORKDIR /lib/test/gtest/
# Set the CMD to your handler
CMD [ "./run_all.x" ]
