cmake_minimum_required(VERSION 2.8)
project( researchProj )
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIR})
# define source files
list(APPEND SRCS
    HumanFallDetection.cpp
    main.cpp
)
# define header files
list(APPEND HDRS
  HumanFallDetection.h
)
add_executable(${PROJECT_NAME} ${SRCS} ${HDRS})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
set(CMAKE_BUILD_TYPE Debug)

