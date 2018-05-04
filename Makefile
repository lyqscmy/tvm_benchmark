# CC = clang
# CXX = clang++
CC = gcc
CXX = g++

CFLAGS = -Wall -O2 -std=c++11 \
	 -I$(HOME)/git/benchmark/include \
	 -I$(HOME)/git/nnvm/tvm/dlpack/include \
	 -I$(HOME)/git/nnvm/tvm/include \
	 -I$(HOME)/git/nnvm/tvm/dmlc-core/include

LDFLAGS = -L$(HOME)/git/benchmark/build/src \
	  -L$(HOME)/git/nnvm/tvm/lib \
	  -ltvm_runtime -lbenchmark -lpthread

all: prog

prog: mlp_bench.cc
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: all clean

clean:
	rm -f $(obj) prog
