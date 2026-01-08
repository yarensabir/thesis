cmake_minimum_required(VERSION 3.10)

# Proje Adı
project(VideoStabilizer)

# C++ Standardı (C++11 veya C++14 uygundur)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV Kütüphanesini Bul
find_package(OpenCV REQUIRED)

# Include Dizinlerini Ekle
include_directories(${OpenCV_INCLUDE_DIRS})

# Kaynak Dosyayı Belirt (src klasörü altındaki main.cpp)
add_executable(Stabilizer src/main.cpp)


# Kütüphaneleri Linkle
target_link_libraries(Stabilizer ${OpenCV_LIBS})

# Derleme sırasında uyarıları göster (Hata ayıklamak için iyidir)
if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-Wall -Wextra)
endif()
