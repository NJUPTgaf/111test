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

#include "source/opt/cooperative_vec_matmul_fallback_pass.h"
#include "source/opt/ir_builder.h"

namespace spvtools {
namespace opt {

Pass::Status CooperativeVecMatMulFallbackPass::Process() {
  bool modified = false;

  for (auto& func : *context()->module()) {
    for (auto& block : func) {
      auto it = block.begin();
      while (it != block.end()) {
        if (it->opcode() ==
            spv::Op::OpCooperativeVectorMatrixMulAddNV) {
          modified |= ExpandInstruction(&block, &it);
          if (it == block.end()) break;
        } else {
          ++it;
        }
      }
    }
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool CooperativeVecMatMulFallbackPass::ExpandInstruction(
    BasicBlock* /*block*/, BasicBlock::iterator* it) {
  Instruction* inst = &**it;
  InstructionBuilder builder(context(), &**it);

  // ---- Extract operands ----
  // Layout: %result = OpCooperativeVectorMatrixMulAddNV %ResultType
  //   %Input, %InputInterp, %Matrix, %MatrixOffset, %MatrixInterp,
  //   %Bias, %BiasOffset, %BiasInterp, %M, %K, %Layout, %Transpose, [%Stride]
  uint32_t input_id = inst->GetSingleWordInOperand(0);
  uint32_t mat_ptr_id = inst->GetSingleWordInOperand(2);
  uint32_t mat_offset_id = inst->GetSingleWordInOperand(3);
  uint32_t bias_ptr_id = inst->GetSingleWordInOperand(5);
  uint32_t bias_offset_id = inst->GetSingleWordInOperand(6);
  uint32_t m_id = inst->GetSingleWordInOperand(8);
  uint32_t k_id = inst->GetSingleWordInOperand(9);
  uint32_t layout_id = inst->GetSingleWordInOperand(10);
  uint32_t transpose_id = inst->GetSingleWordInOperand(11);
  uint32_t stride_id =
      inst->NumInOperands() > 12 ? inst->GetSingleWordInOperand(12) : 0;

  // ---- Resolve constants ----
  uint32_t M = 0, K = 0, mat_byte_offset = 0, bias_byte_offset = 0,
           stride = 0, layout = 0;
  if (!GetConstantUint(m_id, &M) || !GetConstantUint(k_id, &K) ||
      !GetConstantUint(mat_offset_id, &mat_byte_offset) ||
      !GetConstantUint(bias_offset_id, &bias_byte_offset) ||
      !GetConstantUint(layout_id, &layout)) {
    // Non-constant dimensions not supported yet.
    return false;
  }
  if (stride_id != 0 && !GetConstantUint(stride_id, &stride)) {
    return false;
  }
  if (M % kTileSize != 0 || K % kTileSize != 0) {
    // Dimensions not divisible by tile size.
    return false;
  }
  // Only RowMajor, non-transposed supported for now.
  if (layout != 0 /* RowMajor */) return false;

  bool is_transposed = false;
  auto* transpose_def = get_def_use_mgr()->GetDef(transpose_id);
  if (transpose_def && transpose_def->opcode() == spv::Op::OpConstantFalse) {
    is_transposed = false;
  } else if (transpose_def &&
             transpose_def->opcode() == spv::Op::OpConstantTrue) {
    is_transposed = true;
  }
  if (is_transposed) return false;

  // ---- Determine types ----
  // Result type: OpTypeVectorIdEXT %component_type %M
  uint32_t result_type_id = inst->type_id();
  auto* result_type_inst = get_def_use_mgr()->GetDef(result_type_id);
  uint32_t compute_scalar_type =
      result_type_inst->GetSingleWordInOperand(0);  // e.g. %float

  // Matrix buffer element type
  auto* mat_ptr_def = get_def_use_mgr()->GetDef(mat_ptr_id);
  uint32_t mat_ptr_type_id = mat_ptr_def->type_id();
  auto* mat_ptr_type_inst = get_def_use_mgr()->GetDef(mat_ptr_type_id);
  // OpTypePointer: InOperand(0)=storage, InOperand(1)=pointee
  uint32_t mat_pointee_type_id = mat_ptr_type_inst->GetSingleWordInOperand(1);
  auto* mat_pointee_inst = get_def_use_mgr()->GetDef(mat_pointee_type_id);
  // OpTypeRuntimeArray: InOperand(0)=element type
  uint32_t mat_elem_type_id = mat_pointee_inst->GetSingleWordInOperand(0);
  auto* mat_elem_inst = get_def_use_mgr()->GetDef(mat_elem_type_id);
  uint32_t mat_elem_size = 0;
  if (mat_elem_inst->opcode() == spv::Op::OpTypeFloat) {
    mat_elem_size = mat_elem_inst->GetSingleWordInOperand(0) / 8;
  } else if (mat_elem_inst->opcode() == spv::Op::OpTypeInt) {
    mat_elem_size = mat_elem_inst->GetSingleWordInOperand(0) / 8;
  }
  if (mat_elem_size == 0) return false;

  // Bias buffer element type
  auto* bias_ptr_def = get_def_use_mgr()->GetDef(bias_ptr_id);
  uint32_t bias_ptr_type_id = bias_ptr_def->type_id();
  auto* bias_ptr_type_inst = get_def_use_mgr()->GetDef(bias_ptr_type_id);
  uint32_t bias_pointee_type_id = bias_ptr_type_inst->GetSingleWordInOperand(1);
  auto* bias_pointee_inst = get_def_use_mgr()->GetDef(bias_pointee_type_id);
  uint32_t bias_elem_type_id = bias_pointee_inst->GetSingleWordInOperand(0);
  auto* bias_elem_inst = get_def_use_mgr()->GetDef(bias_elem_type_id);
  uint32_t bias_elem_size = 0;
  if (bias_elem_inst->opcode() == spv::Op::OpTypeFloat) {
    bias_elem_size = bias_elem_inst->GetSingleWordInOperand(0) / 8;
  } else if (bias_elem_inst->opcode() == spv::Op::OpTypeInt) {
    bias_elem_size = bias_elem_inst->GetSingleWordInOperand(0) / 8;
  }
  if (bias_elem_size == 0) return false;

  bool mat_needs_convert = (mat_elem_type_id != compute_scalar_type);
  bool bias_needs_convert = (bias_elem_type_id != compute_scalar_type);

  // ---- Ensure required types ----
  uint32_t storage =
      mat_ptr_type_inst->GetSingleWordInOperand(0);
  uint32_t vec4_type_id = EnsureVecNType(compute_scalar_type, kTileSize);
  uint32_t ptr_mat_elem_id =
      EnsurePointerType(mat_elem_type_id, storage);
  uint32_t ptr_bias_elem_id =
      EnsurePointerType(bias_elem_type_id, storage);

  // ---- Tile computation ----
  uint32_t out_tiles = M / kTileSize;
  uint32_t in_tiles = K / kTileSize;

  // Precompute index constants for matrix access
  uint32_t mat_base_idx = mat_byte_offset / mat_elem_size;
  uint32_t mat_row_stride_elems =
      stride != 0 ? stride / mat_elem_size : K;
  uint32_t bias_base_idx = bias_byte_offset / bias_elem_size;

  // Zero constant for accumulator initialization
  uint32_t zero_id =
      GetOrCreateFloatConstant(0.0f, compute_scalar_type);
  std::vector<uint32_t> zero_elems(kTileSize, zero_id);
  uint32_t zero_vec4_id =
      builder.AddCompositeConstruct(vec4_type_id, zero_elems)->result_id();

  std::vector<uint32_t> tile_results(out_tiles);

  for (uint32_t ti = 0; ti < out_tiles; ti++) {
    // acc = <0, 0, 0, 0>
    uint32_t acc_id = zero_vec4_id;

    for (uint32_t tj = 0; tj < in_tiles; tj++) {
      // Extract input tile (vec4) from coopvec value
      uint32_t v_tile_id =
          ExtractInputTile(builder, input_id, tj, vec4_type_id,
                           compute_scalar_type);

      // Load 4 column vectors of the 4x4 matrix tile
      uint32_t col_ids[kTileSize];
      LoadMatTileColumns(builder, mat_ptr_id, mat_base_idx, mat_row_stride_elems,
                         ti, tj, col_ids, vec4_type_id, ptr_mat_elem_id,
                         mat_elem_type_id, compute_scalar_type,
                         mat_needs_convert);

      // Single tile mat-vec multiply
      uint32_t partial_id =
          EmitTileMatVec(builder, col_ids, v_tile_id, vec4_type_id,
                         compute_scalar_type);

      // Accumulate
      acc_id = builder.AddBinaryOp(vec4_type_id, spv::Op::OpFAdd, acc_id,
                                    partial_id)
                   ->result_id();
    }

    // Load bias tile
    uint32_t bias_tile_id =
        LoadBiasTile(builder, bias_ptr_id, bias_base_idx, ti,
                     vec4_type_id, ptr_bias_elem_id, bias_elem_type_id,
                     compute_scalar_type, bias_needs_convert);

    // Add bias
    acc_id = builder.AddBinaryOp(vec4_type_id, spv::Op::OpFAdd, acc_id,
                                  bias_tile_id)
                 ->result_id();

    tile_results[ti] = acc_id;
  }

  // ---- Construct final result ----
  // Collect all M scalar elements from tile results
  std::vector<uint32_t> result_elements;
  for (uint32_t ti = 0; ti < out_tiles; ti++) {
    for (uint32_t i = 0; i < kTileSize; i++) {
      auto* elem = builder.AddCompositeExtract(compute_scalar_type,
                                                tile_results[ti], {i});
      result_elements.push_back(elem->result_id());
    }
  }

  auto* result_inst = builder.AddNaryOp(result_type_id,
                                         spv::Op::OpCompositeConstruct,
                                         result_elements);

  // Replace all uses of the original instruction's result
  context()->ReplaceAllUsesWith(inst->result_id(),
                                result_inst->result_id());

  // Delete original instruction
  *it = (*it).Erase();

  return true;
}

// ===== Single 4x4 tile mat-vec =====
// result = col0*v[0] + col1*v[1] + col2*v[2] + col3*v[3]
uint32_t CooperativeVecMatMulFallbackPass::EmitTileMatVec(
    InstructionBuilder& builder, const uint32_t col_ids[kTileSize],
    uint32_t vec_id, uint32_t vec4_type_id, uint32_t scalar_type_id) {
  std::array<uint32_t, kTileSize> scaled;

  for (uint32_t i = 0; i < kTileSize; i++) {
    // Extract scalar v[i] from vec4
    auto* elem =
        builder.AddCompositeExtract(scalar_type_id, vec_id, {i});
    // col_i * v[i]
    auto* s = builder.AddBinaryOp(vec4_type_id, spv::Op::OpVectorTimesScalar,
                                  col_ids[i], elem->result_id());
    scaled[i] = s->result_id();
  }

  // Tree reduction: 4 vec4 → 1 vec4
  auto* s01 = builder.AddBinaryOp(vec4_type_id, spv::Op::OpFAdd,
                                  scaled[0], scaled[1]);
  auto* s23 = builder.AddBinaryOp(vec4_type_id, spv::Op::OpFAdd,
                                  scaled[2], scaled[3]);
  auto* result = builder.AddBinaryOp(vec4_type_id, spv::Op::OpFAdd,
                                     s01->result_id(), s23->result_id());

  return result->result_id();
}

// ===== Extract a vec4 tile from input coopvec value =====
uint32_t CooperativeVecMatMulFallbackPass::ExtractInputTile(
    InstructionBuilder& builder, uint32_t input_id, uint32_t tile_idx,
    uint32_t vec4_type_id, uint32_t scalar_type_id) {
  std::vector<uint32_t> elems(kTileSize);
  for (uint32_t i = 0; i < kTileSize; i++) {
    uint32_t idx = tile_idx * kTileSize + i;
    auto* elem = builder.AddCompositeExtract(scalar_type_id, input_id, {idx});
    elems[i] = elem->result_id();
  }
  return builder.AddCompositeConstruct(vec4_type_id, elems)->result_id();
}

// ===== Load 4 column vectors of a 4x4 matrix tile from buffer =====
void CooperativeVecMatMulFallbackPass::LoadMatTileColumns(
    InstructionBuilder& builder, uint32_t mat_ptr, uint32_t base_idx,
    uint32_t row_stride_elems, uint32_t row_tile, uint32_t col_tile,
    uint32_t col_ids[kTileSize], uint32_t vec4_type_id,
    uint32_t ptr_elem_type_id, uint32_t mat_elem_type_id,
    uint32_t compute_scalar_type, bool needs_convert) {
  // Load all 16 elements (4x4) with row-major access for locality
  uint32_t elem_ids[kTileSize][kTileSize];
  for (uint32_t r = 0; r < kTileSize; r++) {
    for (uint32_t c = 0; c < kTileSize; c++) {
      uint32_t elem_idx = base_idx +
                          (row_tile * kTileSize + r) * row_stride_elems +
                          (col_tile * kTileSize + c);
      uint32_t idx_id = GetOrCreateIntConstant(elem_idx);
      auto* ptr = builder.AddAccessChain(ptr_elem_type_id, mat_ptr, {idx_id});
      auto* load = builder.AddLoad(mat_elem_type_id, ptr->result_id());
      uint32_t val_id = load->result_id();

      if (needs_convert) {
        auto* conv = builder.AddUnaryOp(compute_scalar_type,
                                         spv::Op::OpFConvert, val_id);
        val_id = conv->result_id();
      }
      elem_ids[r][c] = val_id;
    }
  }

  // Assemble column vectors: col[c] = { elem[0][c], elem[1][c], elem[2][c], elem[3][c] }
  for (uint32_t c = 0; c < kTileSize; c++) {
    std::vector<uint32_t> elems;
    for (uint32_t r = 0; r < kTileSize; r++) {
      elems.push_back(elem_ids[r][c]);
    }
    col_ids[c] =
        builder.AddCompositeConstruct(vec4_type_id, elems)->result_id();
  }
}

// ===== Load a vec4 bias tile from buffer =====
uint32_t CooperativeVecMatMulFallbackPass::LoadBiasTile(
    InstructionBuilder& builder, uint32_t bias_ptr, uint32_t base_idx,
    uint32_t tile_idx, uint32_t vec4_type_id, uint32_t ptr_elem_type_id,
    uint32_t bias_elem_type_id, uint32_t compute_scalar_type,
    bool needs_convert) {
  std::vector<uint32_t> elems(kTileSize);
  for (uint32_t i = 0; i < kTileSize; i++) {
    uint32_t elem_idx = base_idx + tile_idx * kTileSize + i;
    uint32_t idx_id = GetOrCreateIntConstant(elem_idx);
    auto* ptr = builder.AddAccessChain(ptr_elem_type_id, bias_ptr, {idx_id});
    auto* load = builder.AddLoad(bias_elem_type_id, ptr->result_id());
    uint32_t val_id = load->result_id();

    if (needs_convert) {
      auto* conv = builder.AddUnaryOp(compute_scalar_type,
                                       spv::Op::OpFConvert, val_id);
      val_id = conv->result_id();
    }
    elems[i] = val_id;
  }
  return builder.AddCompositeConstruct(vec4_type_id, elems)->result_id();
}

// ===== Type helpers =====

uint32_t CooperativeVecMatMulFallbackPass::EnsureVecNType(
    uint32_t scalar_type_id, uint32_t count) {
  for (auto& inst : context()->module()->types_values()) {
    if (inst.opcode() == spv::Op::OpTypeVector &&
        inst.GetSingleWordInOperand(0) == scalar_type_id &&
        inst.GetSingleWordInOperand(1) == count) {
      return inst.result_id();
    }
  }
  uint32_t id = context()->TakeNextId();
  context()->module()->AddType(std::unique_ptr<Instruction>(
      new Instruction(context(), spv::Op::OpTypeVector, 0, id,
                      {{SPV_OPERAND_TYPE_ID, {scalar_type_id}},
                       {SPV_OPERAND_TYPE_LITERAL_INTEGER, {count}}})));
  return id;
}

uint32_t CooperativeVecMatMulFallbackPass::EnsurePointerType(
    uint32_t type_id, uint32_t storage) {
  for (auto& inst : context()->module()->types_values()) {
    if (inst.opcode() == spv::Op::OpTypePointer &&
        inst.GetSingleWordInOperand(0) == storage &&
        inst.GetSingleWordInOperand(1) == type_id) {
      return inst.result_id();
    }
  }
  uint32_t id = context()->TakeNextId();
  context()->module()->AddType(std::unique_ptr<Instruction>(
      new Instruction(context(), spv::Op::OpTypePointer, 0, id,
      {{SPV_OPERAND_TYPE_STORAGE_CLASS, {storage}},
       {SPV_OPERAND_TYPE_ID, {type_id}}})));
  return id;
}

// ===== Constant helpers =====

uint32_t CooperativeVecMatMulFallbackPass::GetOrCreateIntConstant(
    uint32_t value) {
  // Find or create uint32 type
  uint32_t uint_type_id = 0;
  for (auto& inst : context()->module()->types_values()) {
    if (inst.opcode() == spv::Op::OpTypeInt &&
        inst.GetSingleWordInOperand(0) == 32 &&
        inst.GetSingleWordInOperand(1) == 0) {
      uint_type_id = inst.result_id();
      break;
    }
  }
  if (uint_type_id == 0) {
    uint_type_id = context()->TakeNextId();
    context()->module()->AddType(std::unique_ptr<Instruction>(
        new Instruction(context(), spv::Op::OpTypeInt, 0, uint_type_id,
                        {{SPV_OPERAND_TYPE_LITERAL_INTEGER, {32}},
                         {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0}}})));
  }

  // Find existing constant
  for (auto& inst : context()->module()->types_values()) {
    if (inst.opcode() == spv::Op::OpConstant &&
        inst.type_id() == uint_type_id &&
        inst.GetSingleWordInOperand(0) == value) {
      return inst.result_id();
    }
  }

  // Create new constant
  uint32_t id = context()->TakeNextId();
  context()->module()->AddGlobalValue(std::unique_ptr<Instruction>(
      new Instruction(context(), spv::Op::OpConstant, uint_type_id, id,
                      {{SPV_OPERAND_TYPE_LITERAL_INTEGER, {value}}})));
  return id;
}

uint32_t CooperativeVecMatMulFallbackPass::GetOrCreateFloatConstant(
    float value, uint32_t float_type_id) {
  uint32_t word;
  memcpy(&word, &value, sizeof(word));

  for (auto& inst : context()->module()->types_values()) {
    if (inst.opcode() == spv::Op::OpConstant &&
        inst.type_id() == float_type_id &&
        inst.GetSingleWordInOperand(0) == word) {
      return inst.result_id();
    }
  }

  uint32_t id = context()->TakeNextId();
  context()->module()->AddGlobalValue(std::unique_ptr<Instruction>(
      new Instruction(context(), spv::Op::OpConstant, float_type_id, id,
                      {{SPV_OPERAND_TYPE_LITERAL_INTEGER, {word}}})));
  return id;
}

bool CooperativeVecMatMulFallbackPass::GetConstantUint(uint32_t id,
                                                        uint32_t* out) {
  auto* def = get_def_use_mgr()->GetDef(id);
  if (!def) return false;
  if (def->opcode() == spv::Op::OpConstant) {
    *out = def->GetSingleWordInOperand(0);
    return true;
  }
  return false;
}

}  // namespace opt
}  // namespace spvtools
