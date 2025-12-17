cmake_minimum_required(VERSION 3.23)
project(TaylorTorchBinaryPackage VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SPM_ARTIFACT "Stage a SwiftPM artifact bundle" ON)
set(LIBTORCH_ROOT "" CACHE PATH "Path to a prebuilt libtorch (root containing share/cmake/Torch)")

# --------------------------------------------------------------------------
# Torch setup
# --------------------------------------------------------------------------
if(NOT LIBTORCH_ROOT)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
    set(LIBTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch")
    set(LIBTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch")
  else()
    set(LIBTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
  endif()
endif()

if(EXISTS "${LIBTORCH_ROOT}")
  # libtorch publishes a CMake config in share/cmake/Torch
  list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
else()
  message(FATAL_ERROR "libtorch not found at ${LIBTORCH_ROOT}. Run get-pre-build-torch.sh first or set -DLIBTORCH_ROOT=/path/to/libtorch")
endif()

find_package(Torch REQUIRED)

# SWIFT_BRIDGING_INCLUDE_DIR is set by setenv.sh when using swiftly.
if(DEFINED ENV{SWIFT_BRIDGING_INCLUDE_DIR})
  set(SWIFT_BRIDGING_INCLUDE_DIR "$ENV{SWIFT_BRIDGING_INCLUDE_DIR}")
endif()

# --------------------------------------------------------------------------
# ATenCXX shim library
# --------------------------------------------------------------------------
add_library(ATenCXX STATIC
  ATenCXX/src/aten_shim.cpp
  ATenCXX/include/aten_shim.hpp
  ATenCXX/include/tensor_shim.hpp
  ATenCXX/include/module.modulemap
)
target_include_directories(ATenCXX
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/ATenCXX/include
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(ATenCXX
  PUBLIC
    ${TORCH_LIBRARIES}
)

if(SWIFT_BRIDGING_INCLUDE_DIR)
  target_include_directories(ATenCXX PUBLIC ${SWIFT_BRIDGING_INCLUDE_DIR})
endif()

set_target_properties(ATenCXX PROPERTIES
  OUTPUT_NAME "ATenCXX"
  POSITION_INDEPENDENT_CODE ON
)

install(TARGETS ATenCXX
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
install(DIRECTORY ATenCXX/include/ DESTINATION include)

# --------------------------------------------------------------------------
# SwiftPM artifact bundle (local development)
# --------------------------------------------------------------------------
if(BUILD_SPM_ARTIFACT)
  if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin"))
    message(WARNING "SwiftPM artifact bundling is only wired up for Linux and macOS.")
  else()
    set(SPM_BUNDLE_DIR ${CMAKE_BINARY_DIR}/artifactbundle/ATenCXX.artifactbundle)
    set(SPM_VARIANT_DIR artifacts)

    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
      # Rely on host arch; adjust if you are building universal binaries.
      if(NOT CMAKE_OSX_ARCHITECTURES)
        set(CMAKE_OSX_ARCHITECTURES "${CMAKE_SYSTEM_PROCESSOR}")
      endif()
      set(SPM_TRIPLE "${CMAKE_OSX_ARCHITECTURES}-apple-macosx")
    else()
      if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
        set(SPM_TRIPLE "x86_64-unknown-linux-gnu")
      elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(SPM_TRIPLE "aarch64-unknown-linux-gnu")
      else()
        set(SPM_TRIPLE "${CMAKE_SYSTEM_PROCESSOR}-unknown-linux-gnu")
      endif()
    endif()

    set(SPM_VARIANT_ROOT "${SPM_BUNDLE_DIR}/${SPM_VARIANT_DIR}/${SPM_TRIPLE}")
    set(SPM_LIB_DIR "${SPM_VARIANT_ROOT}/lib")
    set(SPM_INCLUDE_DIR "${SPM_VARIANT_ROOT}/include")

    add_custom_command(
      TARGET ATenCXX POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E rm -f "${SPM_BUNDLE_DIR}.zip"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/artifactbundle"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${SPM_BUNDLE_DIR}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${SPM_VARIANT_ROOT}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${SPM_LIB_DIR}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${SPM_INCLUDE_DIR}"
      COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:ATenCXX>" "${SPM_LIB_DIR}/"
      COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/ATenCXX/include" "${SPM_INCLUDE_DIR}"
      COMMAND ${CMAKE_COMMAND} -E copy_directory "${LIBTORCH_ROOT}/lib" "${SPM_VARIANT_ROOT}/libtorch/lib"
      COMMAND ${CMAKE_COMMAND} -E tar "cfv" "${SPM_BUNDLE_DIR}.zip" --format=zip -C "${CMAKE_BINARY_DIR}/artifactbundle" "ATenCXX.artifactbundle"
      COMMENT "Staging SwiftPM artifact bundle in ${SPM_BUNDLE_DIR}"
    )

    configure_file(
      cmake/info.json.in
      ${SPM_BUNDLE_DIR}/info.json
      @ONLY
    )

    add_custom_target(package-swift
      DEPENDS ATenCXX
      COMMENT "SwiftPM artifact bundle ready at ${SPM_BUNDLE_DIR}"
    )
    add_custom_target(package-swift-zip
      DEPENDS ATenCXX
      COMMENT "SwiftPM artifact bundle zip ready at ${SPM_BUNDLE_DIR}.zip"
    )
  endif()
endif()
