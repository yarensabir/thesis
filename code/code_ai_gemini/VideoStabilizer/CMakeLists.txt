cmake_minimum_required(VERSION 3.10)
project(VideoStabilizer)

set(CMAKE_CXX_STANDARD 11)

# OpenCV paketini bul
find_package(OpenCV REQUIRED)

# Include klasörlerini ekle
include_directories(${OpenCV_INCLUDE_DIRS})

# Çalıştırılabilir dosya (Executable) oluştur
add_executable(Stabilizer src/main.cpp)

# OpenCV kütüphanelerini linkle
target_link_libraries(Stabilizer ${OpenCV_LIBS})