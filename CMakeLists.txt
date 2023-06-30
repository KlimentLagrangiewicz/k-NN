cmake_minimum_required(VERSION 3.20)

project(k-NN LANGUAGES C)

set(CMAKE_C_STANDARD 90)

add_executable(k-NN src/main.c src/help.h src/help.c src/knn.h src/knn.c)

target_link_libraries(k-NN m)

install(TARGETS k-NN
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
