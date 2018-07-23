cmake_minimum_required( VERSION 3.5.1 )
set( CMAKE_CXX_STANDARD 11 )


project( ProjName )

find_package( OpenCV REQUIRED )

file( GLOB SOURCES "src/stitch.cpp" )

add_executable( a.out ${SOURCES} )
target_link_libraries( a.out ${OpenCV_LIBS} )
