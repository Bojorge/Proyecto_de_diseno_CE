cmake_minimum_required(VERSION 3.10)
project(ZeroMQ_message_passing)

set(CMAKE_CXX_STANDARD 11)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZeroMQ REQUIRED libzmq)

include_directories(${ZeroMQ_INCLUDE_DIRS})
link_directories(${ZeroMQ_LIBRARY_DIRS})

add_executable(server server.cpp)
target_link_libraries(server ${ZeroMQ_LIBRARIES})

add_executable(client client.cpp)
target_link_libraries(client ${ZeroMQ_LIBRARIES})
