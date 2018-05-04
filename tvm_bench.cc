#include <benchmark/benchmark.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <fstream>
#include <iterator>

static void BM_MultiThreaded(benchmark::State &state) {
  // Setup code here.

  // tvm module for compiled functions
  tvm::runtime::Module mod_syslib =
      tvm::runtime::Module::LoadFromFile("deploy.so");

  // json graph
  std::ifstream json_in("deploy.json", std::ios::in);
  std::string json_data((std::istreambuf_iterator<char>(json_in)),
                        std::istreambuf_iterator<char>());
  json_in.close();

  // parameters in binary
  std::ifstream params_in("deploy.params", std::ios::binary);
  std::string params_data((std::istreambuf_iterator<char>(params_in)),
                          std::istreambuf_iterator<char>());
  params_in.close();

  // parameters need to be TVMByteArray type to indicate the binary data
  TVMByteArray params_arr;
  params_arr.data = params_data.c_str();
  params_arr.size = params_data.length();

  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;

  // get global function module for graph runtime
  tvm::runtime::Module mod =
      (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(
          json_data, mod_syslib, device_type, device_id);

  DLTensor *x;
  int in_ndim = 2;
  int64_t in_shape[2] = {1000, 1024};
  TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &x);

  // get the function from the module(set input data)
  tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
  set_input("data", x);

  // get the function from the module(load patameters)
  tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
  load_params(params_arr);

  // get the function from the module(run it)
  tvm::runtime::PackedFunc run = mod.GetFunction("run");

  DLTensor *y;
  int out_ndim = 2;
  int64_t out_shape[2] = {1000, 2};
  TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &y);

  // get the function from the module(get output data)
  tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
  get_output(0, y);

  for (auto _ : state) {
    // Run the test as normal.
    run();
  }

  // Teardown code here.
  TVMArrayFree(x);
  TVMArrayFree(y);
}
BENCHMARK(BM_MultiThreaded)->Threads(2)->UseRealTime();
BENCHMARK_MAIN();
