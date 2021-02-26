cmake_minimum_required(VERSION 3.5) 
project(3d_boundary) 
 
#set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
 
find_package(PCL 1.7 REQUIRED) 


include_directories(${PCL_INCLUDE_DIRS})
 
link_directories($PCL_LIBRARY_DIRS)
 
add_definitions(${PCL_DEFINITIONS}) 
 
add_executable(3d_boundary 1_test_boundary.cpp) 
 
target_link_libraries (3d_boundary ${PCL_LIBRARIES}) 
install(TARGETS 3d_boundary RUNTIME DESTINATION bin)
