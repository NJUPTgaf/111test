// Copyright (c) 2025 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SOURCE_OPT_COOPERATIVE_VEC_MATMUL_FALLBACK_PASS_H_
#define SOURCE_OPT_COOPERATIVE_VEC_MATMUL_FALLBACK_PASS_H_

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

class InstructionBuilder;

// Replaces OpCooperativeVectorMatrixMulAddNV with equivalent basic SPIR-V
// operations using 4x4 tiled mat-vec multiply (OpVectorTimesScalar + OpFAdd).
class CooperativeVecMatMulFallbackPass : public Pass {
 public:
  const char* name() const override {
    return "cooperative-vec-matmul-fallback";
  }

  Status Process() override;

 private:
  static constexpr uint32_t kTileSize = 4;

  // Expand a single OpCooperativeVectorMatrixMulAddNV instruction.
  bool ExpandInstruction(BasicBlock* block, BasicBlock::iterator* it);

  // Single 4x4 tile mat-vec: result = sum(col_i * v[i], i=0..3)
  uint32_t EmitTileMatVec(InstructionBuilder& builder,
                           const uint32_t* col_ids,
                           uint32_t vec_id, uint32_t vec4_type_id,
                           uint32_t scalar_type_id);

  // Extract a vec4 tile from input coopvec value (elements [tile_idx*4 .. tile_idx*4+3]).
  uint32_t ExtractInputTile(InstructionBuilder& builder, uint32_t input_id,
                             uint32_t tile_idx, uint32_t vec4_type_id,
                             uint32_t scalar_type_id);

  // Load 4 column vectors of a 4x4 matrix tile from a buffer.
  void LoadMatTileColumns(InstructionBuilder& builder, uint32_t mat_ptr,
                           uint32_t base_idx, uint32_t row_stride_elems,
                           uint32_t row_tile, uint32_t col_tile,
                           uint32_t* col_ids,
                           uint32_t vec4_type_id,
                           uint32_t ptr_elem_type_id,
                           uint32_t mat_elem_type_id,
                           uint32_t compute_scalar_type,
                           bool needs_convert);

  // Load a vec4 bias tile from a buffer.
  uint32_t LoadBiasTile(InstructionBuilder& builder, uint32_t bias_ptr,
                         uint32_t base_idx, uint32_t tile_idx,
                         uint32_t vec4_type_id,
                         uint32_t ptr_elem_type_id,
                         uint32_t bias_elem_type_id,
                         uint32_t compute_scalar_type,
                         bool needs_convert);

  // Type helpers
  uint32_t EnsureVecNType(uint32_t scalar_type_id, uint32_t count);
  uint32_t EnsurePointerType(uint32_t type_id, uint32_t storage);
  uint32_t GetOrCreateIntConstant(uint32_t value);
  uint32_t GetOrCreateFloatConstant(float value, uint32_t float_type_id);

  // Resolve an IdRef to a compile-time uint32_t constant value. Returns false
  // if the id does not resolve to a constant.
  bool GetConstantUint(uint32_t id, uint32_t* out);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_COOPERATIVE_VEC_MATMUL_FALLBACK_PASS_H_
