#include "aten_shim.hpp"
#include "tensor_shim.hpp"

// Keep a tiny symbol so the archive is never empty even if all functions are
// inlined by the compiler.
extern "C" int aten_cxx_placeholder_symbol() {
  return 0;
}
