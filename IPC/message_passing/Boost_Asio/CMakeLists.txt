cmake_minimum_required(VERSION 3.10)
project(BoostAsio_message_passing)

set(CMAKE_CXX_STANDARD 11)

find_package(Boost REQUIRED COMPONENTS system)

include_directories(${Boost_INCLUDE_DIRS})

add_executable(server server.cpp)
target_link_libraries(server ${Boost_LIBRARIES})

add_executable(client client.cpp)
target_link_libraries(client ${Boost_LIBRARIES})
