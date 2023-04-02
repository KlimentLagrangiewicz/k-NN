cmake_minimum_required(VERSION 3.5)

project(k-NN LANGUAGES C)

set(CMAKE_C_STANDARD 90)

add_executable(k-NN main.c help.h help.c knn.h knn.c)

target_link_libraries(k-NN m)

install(TARGETS k-NN
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
