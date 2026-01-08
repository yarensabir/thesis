cmake_minimum_required(VERSION 3.10)

# Proje Adı
project(VideoStabilizer)

# C++ Standardı
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV Kütüphanesini Bul
find_package(OpenCV REQUIRED)

# Include Dizinlerini Ekle
include_directories(${OpenCV_INCLUDE_DIRS})

# src klasöründeki TÜM .cpp dosyalarını bul
file(GLOB SOURCES "src/*.cpp")

# Bulunan tüm kaynak dosyaları ile executable oluştur
add_executable(Stabilizer ${SOURCES})

# Kütüphaneleri Linkle
target_link_libraries(Stabilizer ${OpenCV_LIBS})

# Uyarıları aç
if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-Wall -Wextra)
endif()