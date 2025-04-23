cmake_minimum_required(VERSION 3.5)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(my_cool_project)
find_package(OpenCV REQUIRED COMPONENTS core aruco highgui videoio imgproc)
add_executable(my_cool_app)
target_sources(my_cool_app PRIVATE main.cpp)
target_include_directories(my_cool_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(my_cool_app PRIVATE ${OpenCV_LIBS})

#
#


#add_executable(cw_marker_creator)
#target_sources(cw_marker_creator PRIVATE main_marker_creator.cpp )
#target_include_directories(my_cool_app PRIVATE ${OpenCV_INCLUDE_DIRS})
#target_link_libraries(my_cool_app PRIVATE ${OpenCV_LIBS})




