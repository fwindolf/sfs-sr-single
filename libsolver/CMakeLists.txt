SET(LIB_NAME solver)

FILE(GLOB INCLUDES "include/${LIB_NAME}/*.h")
FILE(GLOB SOURCES  "src/**/*.cpp" "src/*.cpp")
FILE(GLOB CUDA_SOURCES  "src/*.cu" "src/**/*.cu")

#MESSAGE("${LIB_NAME} Includes: ${INCLUDES}")
#MESSAGE("${LIB_NAME} Sources: ${SOURCES}")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

ADD_LIBRARY(${LIB_NAME} STATIC ${INCLUDES} ${SOURCES} ${CUDA_SOURCES})


SET(LIBRARIES 
    ${Opt_LIBRARIES}
    ${Terra_LIBRARIES}
    cusolver
    cublas
    cusparse
    dl
    z
    tinfo
    core
    image
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

    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

    TARGET_LINK_LIBRARIES(${TEST_EXECUTABLE}
        gtest gtest_main 
        pthread 
        ${OpenCV_LIBS}
        ${LIBRARIES}
        ${LIB_NAME}
    )

    SET(TEST_ALBEDO   0)
    SET(TEST_LIGHT    0)
    SET(TEST_THETA    1)
    SET(TEST_DEPTH    0)
    SET(TEST_INPAINT  0)
    SET(TEST_PARTS    1)
    SET(TEST_COMBINED 1)

    ADD_DEFINITIONS(
        -DTEST_ALBEDO=${TEST_ALBEDO}
        -DTEST_LIGHT=${TEST_LIGHT}
        -DTEST_THETA=${TEST_THETA}
        -DTEST_DEPTH=${TEST_DEPTH}
        -DTEST_INPAINT=${TEST_INPAINT}
        -DTEST_PARTS=${TEST_PARTS}
        -DTEST_COMBINED=${TEST_COMBINED}
    )

    ADD_TEST(${TEST_EXECUTABLE} 
        test
    )

ENDIF()
