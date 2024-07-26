cmake_minimum_required(VERSION 3.10)
project(POCO_Sockets)

set(CMAKE_CXX_STANDARD 11)

find_package(Poco REQUIRED Net)

include_directories(${Poco_INCLUDE_DIRS})

add_executable(server server.cpp)
target_link_libraries(server ${Poco_LIBRARIES})

add_executable(client client.cpp)
target_link_libraries(client ${Poco_LIBRARIES})
