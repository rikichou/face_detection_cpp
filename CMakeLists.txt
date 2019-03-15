cmake_minimum_required(VERSION 3.5.1)

project(demo)

set(CMAKE_CXX_STANDARD 11)

set(TENSORFLOW_DIR /home/tensortec/riki/workspace/pro/object_detection/source/server/tensorflow)

include_directories(${TENSORFLOW_DIR})
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/proto)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/protobuf-host/include)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/eigen)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/nsync/public)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/absl)

link_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/lib)
link_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/protobuf-host/lib)
link_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11)
link_directories(${TENSORFLOW_DIR}/bazel-bin/tensorflow)

add_executable(demo main.cpp face_detecion.cpp)

target_link_libraries(demo tensorflow_cc tensorflow_framework opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs)