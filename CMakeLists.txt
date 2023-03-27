# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.20)
project(nsight_demo)

# Enable cuda language
set(CMAKE_CUDA_ARCHITECTURES "70;80")
enable_language(CUDA)

# Download MatX
include(FetchContent)
set(MATX_EN_FILEIO ON)
set(MATX_EN_CUTENSOR ON)
set(MATX_NVTX_FLAGS ON)

FetchContent_Declare(
  MatX
  GIT_REPOSITORY https://github.com/NVIDIA/MatX.git
  GIT_TAG main
)
FetchContent_MakeAvailable(MatX)

add_executable(nsight-demo
  src/main.cu
)
target_include_directories(nsight-demo PRIVATE
  include/
)
target_link_libraries(nsight-demo
  PRIVATE matx::matx -lnvToolsExt
)
target_compile_options(nsight-demo PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
>)
set_target_properties(${TARGET}
  PROPERTIES
    LINKER_LANGUAGE             CUDA
    CUDA_SEPARABLE_COMPILATION  ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)