#!/bin/bash

# Save current path
SOURCE_DIR=$(pwd)


# Check Prerequisites
## Clang
INSTALL_CLANG=1
clang_version=$(clang++ --version | grep -E "version")
if [[ -n "${clang_version// /}" ]]; 
then    
    re=".* version ([0-9])\.([0-9]).*"
    if [[ "$clang_version" =~ $re ]] 
    then
        if [ "${BASH_REMATCH[1]}" -eq "6" ]
        then
            echo "Found clang-${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
            INSTALL_CLANG=0
        fi
    fi
fi

INSTALL_LLVM=1
llvm_version=$(llvm-config --version)
re="([0-9])\.([0-9]).*"
if [[ "$llvm_version" =~ $re ]] 
then
    if [ "${BASH_REMATCH[1]}" -eq "6" ]
    then
        echo "Found llvm-${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
        INSTALL_LLVM=0
    fi
fi

INSTALL_CUDA=1
nvcc_version=$(nvcc --version | grep -oE "V[0-9].[0-9]+.[0-9]+")
if [[ -n "${nvcc_version// /}" ]]; 
then 
    re="([0-9])\.([0-9]).*"
    if [[ "$nvcc_version" =~ $re ]];
    then
        if [ "${BASH_REMATCH[1]}" -ge "9" ]
        then
            echo "Found cuda-${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
            INSTALL_CUDA=0
        fi
    fi
fi

if [ -f "$INSTALL_CLANG" ]
then
    echo "Install clang locally: Use http://releases.llvm.org/download.html#6.0.1 for your platform"
fi

if [ -f "$INSTALL_LLVM" ]
then
    echo "Install llvm locally: Use http://releases.llvm.org/download.html#6.0.1 for your platform"
fi

if [ -f "$INSTALL_CUDA" ]
then
    echo "Install cuda locally: Use https://developer.nvidia.com/cuda-90-download-archive for your platform"
fi

# Init submodules
git submodule update --init

# Setup Pangolin
cd third_party/Pangolin
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_BUILD_TYPE=Release -DCC=/usr/bin/clang -DCXX=/usr/bin/clang++
make -j8 install
cd $SOURCE_DIR

# Setup cuda-image
cd third_party/cuda-image
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED=ON -DPANGOLIN_DIR=$SOURCE_DIR/third_party/Pangolin -DEIGEN_DIR=$SOURCE_DIR/third_party/Eigen -DEIGEN3_INCLUDE_DIR=$SOURCE_DIR/third_party/Eigen/include -DCC=/usr/bin/clang -DCXX=/usr/bin/clang++
make -j8 install
cd $SOURCE_DIR

# Setup Opt
cd third_party

## Download terra
wget -N https://github.com/zdevito/terra/releases/download/release-1.0.0-beta1/terra-Linux-x86_64-2e2032e.zip
unzip -q -o terra-Linux-x86_64-2e2032e.zip 
ln -s terra-Linux-x86_64-2e2032e terra
rm terra-Linux-x86_64-2e2032e.zip

# Setup Opt
cd Opt/API 
make
cd $SOURCE_DIR

# Show instructions
echo "--------------------------------------------------"
echo " Finished installing environment, build Apps now"
echo "--------------------------------------------------"
echo "mkdir build"
echo "cd build"
echo "cmake -DCC=/usr/bin/clang -DCXX=/usr/bin/clang++ .. && make -j8"
echo "./bin/AppSfs -h"
echo "--------------------------------------------------"



