SET(LIB_NAME image)

FILE(GLOB INCLUDES "include/${LIB_NAME}/*.h" 
                   "include/${LIB_NAME}/**/*.h"
    )

FILE(GLOB SOURCES  "src/*.cpp"
                   "src/**/*.cpp"
                   "src/**/**/*.cpp"
                   "src/**/*.cu"
                   "src/**/**/*.cu"
    )

#MESSAGE("${LIB_NAME} Includes: ${INCLUDES}")
#MESSAGE("${LIB_NAME} Sources: ${SOURCES}")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../libsolver/include)

ADD_LIBRARY(${LIB_NAME} STATIC ${INCLUDES} ${SOURCES})

SET(LIBRARIES 
    cuimage
    pangolin
    glut
    GLU
    GL
    GLEW
    core 
    solver
)


TARGET_LINK_LIBRARIES(${LIB_NAME}
    ${LIBRARIES}
)

SET_TARGET_PROPERTIES(${LIB_NAME}
    PROPERTIES LINKER_LANGUAGE CXX
    # COMPILE_FLAGS
    # LINK_FLAGS
)

# ------------------------------------------------------------------------
# tests
IF(BUILD_TESTS)

    SET(TEST_EXECUTABLE ${LIB_NAME}_test)
    FILE(GLOB TEST_SOURCES test/test_*.cpp)

    ADD_EXECUTABLE(${TEST_EXECUTABLE}
        test/run.cpp
        ${TEST_SOURCES}
    )

    TARGET_LINK_LIBRARIES(${TEST_EXECUTABLE}
        gtest gtest_main 
        pthread 
        ${OpenCV_LIBS}
        ${LIBRARIES}
        ${LIB_NAME}
    )

    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)


    ADD_TEST(${TEST_EXECUTABLE} 
        test
    )
    
ENDIF()
