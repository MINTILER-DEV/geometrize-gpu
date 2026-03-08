cmake_minimum_required(VERSION 3.16)
project(geometrize LANGUAGES CXX)

option(GEOMETRIZE_ENABLE_OPENGL_COMPUTE "Enable OpenGL compute shader candidate evaluation" OFF)

file(GLOB_RECURSE GEOMETRIZE_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/geometrize/geometrize/*.h")
file(GLOB_RECURSE GEOMETRIZE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/geometrize/geometrize/*.cpp")

add_library(geometrize STATIC ${GEOMETRIZE_HEADERS} ${GEOMETRIZE_SOURCES})
target_include_directories(geometrize PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/geometrize")
target_compile_features(geometrize PUBLIC cxx_std_17)

if(GEOMETRIZE_ENABLE_OPENGL_COMPUTE)
    find_package(OpenGL REQUIRED)
    find_package(GLEW REQUIRED)

    target_compile_definitions(geometrize PUBLIC GEOMETRIZE_ENABLE_OPENGL_COMPUTE)
    target_link_libraries(geometrize PUBLIC OpenGL::GL GLEW::GLEW)
endif()

set(GEOMETRIZE_SHADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/evaluate_shapes.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/error_reduce.comp")
set(GEOMETRIZE_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")

add_custom_target(geometrize_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GEOMETRIZE_SHADER_OUTPUT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GEOMETRIZE_SHADER_FILES} "${GEOMETRIZE_SHADER_OUTPUT_DIR}"
    COMMENT "Copying Geometrize compute shaders")

add_dependencies(geometrize geometrize_shaders)

install(TARGETS geometrize)
install(FILES ${GEOMETRIZE_SHADER_FILES} DESTINATION share/geometrize/shaders)
