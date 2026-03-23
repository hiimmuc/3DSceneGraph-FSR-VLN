cmake_minimum_required(VERSION 3.8)
project(g1_move)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
#find_package(tf2_ros REQUIRED)
#find_package(tf2 REQUIRED)
find_package(unitree_sdk2)

# Create executables
add_executable(g1_getvel_node src/getvel.cpp)
#target_link_libraries(g1_getvel_node unitree_sdk2)
# Link dependencies
ament_target_dependencies(g1_getvel_node
  rclcpp
  nav_msgs
  geometry_msgs
  rosidl_typesupport_introspection_cpp
#  tf2_ros
#  tf2
)
add_executable(g1_pubvel_node src/pubvel.cpp)
target_link_libraries(g1_pubvel_node unitree_sdk2)
# Install executables
install(TARGETS g1_getvel_node 
  DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS g1_pubvel_node 
  DESTINATION lib/${PROJECT_NAME}
)

# Export dependencies
ament_export_dependencies(
  rclcpp
  nav_msgs
  geometry_msgs
  tf2_ros
  tf2
)

ament_package()
