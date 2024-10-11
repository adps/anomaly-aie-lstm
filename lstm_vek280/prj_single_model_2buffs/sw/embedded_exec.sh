#!/bin/bash
export XILINX_XRT=/usr
dmesg -n 4 && echo "Hide DRM messages..."
cd /run/media/*1
./host.exe a.xclbin