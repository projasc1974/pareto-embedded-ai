cmake_minimum_required(VERSION 3.13)

include(pico_sdk_import.cmake)

project(pico_fashion_rlm_naive C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

add_executable(pico_fashion_rlm_naive
    main.c
)

target_compile_options(pico_fashion_rlm_naive PRIVATE
    -O3
    -fno-math-errno
    -ffunction-sections
    -fdata-sections
)
target_link_options(pico_fashion_rlm_naive PRIVATE
    -Wl,--gc-sections
)

target_include_directories(pico_fashion_rlm_naive PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(pico_fashion_rlm_naive
    pico_stdlib
    hardware_timer
)

pico_enable_stdio_usb(pico_fashion_rlm_naive 1)
pico_enable_stdio_uart(pico_fashion_rlm_naive 0)

pico_add_extra_outputs(pico_fashion_rlm_naive)
