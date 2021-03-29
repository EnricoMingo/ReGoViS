cmake_minimum_required(VERSION 3.0)

project(ReGoVis)

find_package(cartesian_interface)

find_package(catkin REQUIRED COMPONENTS cartesian_interface
                                        urdf
                                        rospy
                                        std_msgs
                                        message_generation)

find_package(cartesian_interface REQUIRED)

catkin_python_setup()

generate_messages(
   DEPENDENCIES
   std_msgs
)

catkin_package(

    INCLUDE_DIRS

    LIBRARIES
   
    CATKIN_DEPENDS

    )


include_directories(include
    ${catkin_INCLUDE_DIRS}
)


catkin_install_python(PROGRAMS python/reference_governor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

