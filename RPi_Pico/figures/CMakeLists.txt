cmake_minimum_required(VERSION 3.5)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# Nombre del proyecto
project(tinyml_naive)

# Opciones de compilación para optimización (ninguna - naive)
idf_build_set_property(COMPILE_OPTIONS "-O0;-g" APPEND)

# Habilitar math.h
set(EXTRA_COMPONENT_DIRS)
