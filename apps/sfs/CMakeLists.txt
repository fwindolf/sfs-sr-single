SET(SFS_LIB_DEPENDENCIES
    ${OpenCV_LIBS}
    ${Opt_LIBRARIES}
    ${Terra_LIBRARIES}
    dl
    z
    tinfo
    core 
    image
    solver
)

FILE(GLOB INCLUDES "include/*.h" "include/**/*.h")
FILE(GLOB SOURCES  "src/*.cpp" "src/**/*.cpp" "src/*.cu" "src/**/*.cu")
SET(MAIN "main.cpp")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

SET(APP_SFS AppSfs)
CUDA_ADD_EXECUTABLE(${APP_SFS}
    ${MAIN}
    ${SOURCES}
    ${INCLUDES} 
)
TARGET_LINK_LIBRARIES(${APP_SFS}
    ${SFS_LIB_DEPENDENCIES}
)

TARGET_COMPILE_OPTIONS(${APP_SFS}
	PRIVATE -std=c++11
)

