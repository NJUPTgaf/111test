#ifndef SOURCE_OPT_COOPERATIVE_VEC_MATMUL_FALLBACK_PASS_H_
#define SOURCE_OPT_COOPERATIVE_VEC_MATMUL_FALLBACK_PASS_H_

#include "source/opt/pass.h"

namespace spvtools
{
  namespace opt
  {

    class CooperativeVecMatMulFallbackPass : public Pass
    {
    public:
      // 自定义指令的 opcode，按实际值替换
      static constexpr SpvOp kTargetOpcode = static_cast<SpvOp>(/* your opcode */);

      const char *name() const override
      {
        return "cooperative-vec-matmul-fallback";
      }

      Status Process() override;

    private:
      static constexpr uint32_t kTileSize = 4;

      // 展开单条自定义指令
      bool ExpandInstruction(BasicBlock *block, BasicBlock::iterator *it);

      // 核心：单个 tile 的 mat×vec，传入 4 个列向量 id，使用 OpVectorTimesScalar
      // result = col0*v[0] + col1*v[1] + col2*v[2] + col3*v[3]
      uint32_t EmitTileMatVec(InstructionBuilder &builder,
                              const uint32_t col_ids[kTileSize],
                              uint32_t vec_id,
                              uint32_t vec4_type_id,
                              uint32_t scalar_type_id);

      // 从 array<float, N> 中加载 tile_idx*4 .. tile_idx*4+3，构造 vec4
      uint32_t LoadVec4Tile(InstructionBuilder &builder,
                            uint32_t base_ptr, uint32_t tile_idx,
                            uint32_t vec4_type_id,
                            uint32_t ptr_float_type_id,
                            uint32_t float_type_id);

      // 从行主序 array<array<float,N>,M> 中加载 4x4 tile 的 4 个列向量
      // 列 c = { M[r0][c], M[r1][c], M[r2][c], M[r3][c] }，存入 col_ids[c]
      void LoadMatTileColumns(InstructionBuilder &builder,
                              uint32_t base_ptr,
                              uint32_t row_tile, uint32_t col_tile,
                              uint32_t col_ids[kTileSize],
                              uint32_t vec4_type_id,
                              uint32_t ptr_float_type_id,
                              uint32_t float_type_id);

      // 将 vec4 拆存到 array<float, N> 的 tile_idx*4 .. tile_idx*4+3
      void StoreVec4Tile(InstructionBuilder &builder,
                         uint32_t base_ptr, uint32_t tile_idx,
                         uint32_t val_id,
                         uint32_t ptr_float_type_id,
                         uint32_t float_type_id);

      // 类型辅助
      uint32_t EnsureVec4Type(uint32_t float_type_id);
      uint32_t EnsurePointerType(uint32_t type_id, SpvStorageClass storage);

      // 常量辅助：获取或创建 uint32 常量
      uint32_t GetOrCreateIntConstant(uint32_t value);

      // 从指针指令获取 pointee 类型 id
      // OpTypePointer: InOperand(0)=storage_class, InOperand(1)=pointee_type_id
      uint32_t GetPointeeTypeId(Instruction *ptr_inst);
    };

  } // namespace opt
} // namespace spvtools

#endif
