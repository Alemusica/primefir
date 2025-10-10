cmake_minimum_required(VERSION 3.21)
project(primefir CXX)

# macOS + standard
set(CMAKE_OSX_ARCHITECTURES "arm64")
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0")
set(CMAKE_CXX_STANDARD 17)

# Percorso Max SDK come il resto del progetto
if(NOT DEFINED MAX_SDK_PATH)
  set(MAX_SDK_PATH "$ENV{HOME}/max-sdk")
endif()

# Usa gli script del Max SDK: impostano flags/defines/linker corretti (fondamentale!)
include("${MAX_SDK_PATH}/source/max-sdk-base/script/max-pretarget.cmake")

# Sorgenti
file(GLOB SRC
  "${CMAKE_CURRENT_LIST_DIR}/source/*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/source/*.cc"
  "${CMAKE_CURRENT_LIST_DIR}/source/*.cxx"
)

# Skip any backup sources so they don't get compiled alongside the real one.
list(FILTER SRC EXCLUDE REGEX "_backup")

# Target SENZA tilde; bundle CON tilde
add_library(primefir MODULE ${SRC})
set_target_properties(primefir PROPERTIES
  BUNDLE TRUE
  BUNDLE_EXTENSION "mxo"
  OUTPUT_NAME "primefir~"
)

# Include di c74support (il pretarget espone C74_INCLUDES se presente)
if(DEFINED C74_INCLUDES)
  target_include_directories(primefir PRIVATE ${C74_INCLUDES})
endif()

# ⚠️ NON ridefinire C74_X64: lo setta già il Max SDK (il tuo warning veniva da lì)
# target_compile_definitions(primefir PRIVATE C74_X64)  # <-- via!

# Post-target del Max SDK: imposta link a MaxAPI/framework ecc.
include("${MAX_SDK_PATH}/source/max-sdk-base/script/max-posttarget.cmake")

# Install nella stessa cartella degli altri externals
if(NOT DEFINED MAX_PACKAGE_DIR)
  set(MAX_PACKAGE_DIR "$ENV{HOME}/Documents/Max 9/Packages/MyDev/externals")
endif()

install(TARGETS primefir
  BUNDLE  DESTINATION "${MAX_PACKAGE_DIR}"
  LIBRARY DESTINATION "${MAX_PACKAGE_DIR}")
