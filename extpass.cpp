#include "source/opt/cooperative_vec_matmul_fallback_pass.h"
#include "source/opt/ir_builder.h"

namespace spvtools
{
  namespace opt
  {

    Pass::Status CooperativeVecMatMulFallbackPass::Process()
    {
      bool modified = false;

      for (auto &func : *context()->module())
      {
        for (auto &block : func)
        {
          auto it = block.begin();
          while (it != block.end())
          {
            if (it->opcode() == kTargetOpcode)
            {
              modified |= ExpandInstruction(&block, &it);
              if (it == block.end())
                break;
            }
            else
            {
              ++it;
            }
          }
        }
      }

      return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
    }

    // ===== 核心：单个 tile 的 mat4x4 × vec4 =====
    // 列分解：result = col0*v[0] + col1*v[1] + col2*v[2] + col3*v[3]
    // 每步 col_i * v[i] = OpVectorTimesScalar(col_i, v[i])
    uint32_t CooperativeVecMatMulFallbackPass::EmitTileMatVec(
        InstructionBuilder &builder,
        const uint32_t col_ids[kTileSize],
        uint32_t vec_id,
        uint32_t vec4_type_id,
        uint32_t scalar_type_id)
    {
      std::array<uint32_t, kTileSize> scaled;

      for (uint32_t i = 0; i < kTileSize; i++)
      {
        // 提取向量第 i 个元素 (scalar)
        auto *elem =
            builder.AddCompositeExtract(scalar_type_id, vec_id, {i});
        // col_i * v[i] -> vec4
        auto *s = builder.AddBinaryOp(vec4_type_id, SpvOpVectorTimesScalar,
                                      col_ids[i], elem->result_id());
        scaled[i] = s->result_id();
      }

      // 树形归约：4 个 vec4 相加
      auto *s01 = builder.AddBinaryOp(vec4_type_id, SpvOpFAdd,
                                      scaled[0], scaled[1]);
      auto *s23 = builder.AddBinaryOp(vec4_type_id, SpvOpFAdd,
                                      scaled[2], scaled[3]);
      auto *result = builder.AddBinaryOp(vec4_type_id, SpvOpFAdd,
                                         s01->result_id(), s23->result_id());

      return result->result_id();
    }

    // ===== 主展开函数：按 4x4 分块计算 =====
    // coopvecTest<float,N>  -> OpTypeArray %float %N
    // coopmatTest<float,M,N> -> OpTypeArray(OpTypeArray %float %N) %M, 行主序
    bool CooperativeVecMatMulFallbackPass::ExpandInstruction(
        BasicBlock *block, BasicBlock::iterator *it)
    {
      Instruction *inst = &**it;
      InstructionBuilder builder(context(), &**it);

      // 操作数: outV(ptr), inV(ptr), inM(ptr), b(ptr)
      uint32_t outV_ptr = inst->GetSingleWordInOperand(0);
      uint32_t inV_ptr = inst->GetSingleWordInOperand(1);
      uint32_t inM_ptr = inst->GetSingleWordInOperand(2);
      uint32_t b_ptr = inst->GetSingleWordInOperand(3);

      // ---- 类型推导 ----
      // coopvecTest<float,N> = OpTypeArray %float %N
      auto *inV_def = get_def_use_mgr()->GetDef(inV_ptr);
      uint32_t inV_pointee = GetPointeeTypeId(inV_def);
      auto *inV_type_inst = get_def_use_mgr()->GetDef(inV_pointee);

      uint32_t float_type_id = inV_type_inst->GetSingleWordInOperand(0);
      uint32_t N_const_id = inV_type_inst->GetSingleWordInOperand(1);
      uint32_t N = get_def_use_mgr()->GetDef(N_const_id)->GetSingleWordInOperand(0);

      // coopmatTest<float,M,N> = OpTypeArray %row_type %M
      // row_type = OpTypeArray %float %N
      auto *inM_def = get_def_use_mgr()->GetDef(inM_ptr);
      uint32_t inM_pointee = GetPointeeTypeId(inM_def);
      auto *inM_type_inst = get_def_use_mgr()->GetDef(inM_pointee);

      uint32_t M_const_id = inM_type_inst->GetSingleWordInOperand(1);
      uint32_t M = get_def_use_mgr()->GetDef(M_const_id)->GetSingleWordInOperand(0);

      // ---- 确保所需类型存在 ----
      uint32_t vec4_type_id = EnsureVec4Type(float_type_id);

      // 推导指针的 storage class
      auto *ptr_type_inst = get_def_use_mgr()->GetDef(inV_def->type_id());
      SpvStorageClass storage = static_cast<SpvStorageClass>(
          ptr_type_inst->GetSingleWordInOperand(0));

      uint32_t ptr_float_id = EnsurePointerType(float_type_id, storage);

      // ---- 分块计算 ----
      uint32_t out_tiles = M / kTileSize;
      uint32_t in_tiles = N / kTileSize;

      for (uint32_t ti = 0; ti < out_tiles; ti++)
      {
        uint32_t acc_id = 0;

        for (uint32_t tj = 0; tj < in_tiles; tj++)
        {
          // 加载输入向量 tile (vec4)
          uint32_t v_id = LoadVec4Tile(builder, inV_ptr, tj,
                                       vec4_type_id, ptr_float_id,
                                       float_type_id);

          // 加载矩阵 tile 的 4 个列向量 (从行主序数据中提取列)
          uint32_t col_ids[kTileSize];
          LoadMatTileColumns(builder, inM_ptr, ti, tj, col_ids,
                             vec4_type_id, ptr_float_id, float_type_id);

          // 计算单个 tile 的 mat×vec
          uint32_t partial_id = EmitTileMatVec(builder, col_ids, v_id,
                                               vec4_type_id, float_type_id);

          // 累加
          if (acc_id == 0)
          {
            acc_id = partial_id;
          }
          else
          {
            acc_id = builder.AddBinaryOp(vec4_type_id, SpvOpFAdd,
                                         acc_id, partial_id)
                         ->result_id();
          }
        }

        // 加 bias tile
        uint32_t bias_id = LoadVec4Tile(builder, b_ptr, ti,
                                        vec4_type_id, ptr_float_id,
                                        float_type_id);
        acc_id = builder.AddBinaryOp(vec4_type_id, SpvOpFAdd,
                                     acc_id, bias_id)
                     ->result_id();

        // 存储结果 tile
        StoreVec4Tile(builder, outV_ptr, ti, acc_id, ptr_float_id, float_type_id);
      }

      // 删除原始指令
      *it = inst->RemoveFromList();
      delete inst;

      return true;
    }

    // ===== Tile 加载/存储 =====

    // 从 array<float, N> 中加载 tile_idx*4 .. tile_idx*4+3，构造 vec4
    uint32_t CooperativeVecMatMulFallbackPass::LoadVec4Tile(
        InstructionBuilder &builder,
        uint32_t base_ptr, uint32_t tile_idx,
        uint32_t vec4_type_id,
        uint32_t ptr_float_type_id,
        uint32_t float_type_id)
    {
      std::vector<uint32_t> elems(kTileSize);
      for (uint32_t i = 0; i < kTileSize; i++)
      {
        uint32_t idx_id = GetOrCreateIntConstant(tile_idx * kTileSize + i);
        auto *ptr = builder.AddAccessChain(ptr_float_type_id,
                                           base_ptr, {idx_id});
        auto *load = builder.AddLoad(float_type_id, ptr->result_id());
        elems[i] = load->result_id();
      }
      return builder.AddCompositeConstruct(vec4_type_id, elems)->result_id();
    }

    // 从行主序 array<array<float,N>,M> 中提取 4x4 tile 的 4 个列向量
    // 列 c = { M[r0][c], M[r1][c], M[r2][c], M[r3][c] } -> vec4
    // 加载顺序按行主序 (r 外层, c 内层) 以获得更好的内存局部性
    void CooperativeVecMatMulFallbackPass::LoadMatTileColumns(
        InstructionBuilder &builder,
        uint32_t base_ptr,
        uint32_t row_tile, uint32_t col_tile,
        uint32_t col_ids[kTileSize],
        uint32_t vec4_type_id,
        uint32_t ptr_float_type_id,
        uint32_t float_type_id)
    {
      // 按行主序加载所有 16 个元素
      uint32_t elem_ids[kTileSize][kTileSize];
      for (uint32_t r = 0; r < kTileSize; r++)
      {
        for (uint32_t c = 0; c < kTileSize; c++)
        {
          uint32_t row_idx = GetOrCreateIntConstant(row_tile * kTileSize + r);
          uint32_t col_idx = GetOrCreateIntConstant(col_tile * kTileSize + c);
          auto *ptr = builder.AddAccessChain(ptr_float_type_id, base_ptr,
                                             {row_idx, col_idx});
          auto *load = builder.AddLoad(float_type_id, ptr->result_id());
          elem_ids[r][c] = load->result_id();
        }
      }

      // 组装列向量: col_ids[c] = { elem[0][c], elem[1][c], elem[2][c], elem[3][c] }
      for (uint32_t c = 0; c < kTileSize; c++)
      {
        std::vector<uint32_t> elems;
        for (uint32_t r = 0; r < kTileSize; r++)
        {
          elems.push_back(elem_ids[r][c]);
        }
        col_ids[c] =
            builder.AddCompositeConstruct(vec4_type_id, elems)->result_id();
      }
    }

    // 将 vec4 拆存到 array<float, N> 的 tile_idx*4 .. tile_idx*4+3
    void CooperativeVecMatMulFallbackPass::StoreVec4Tile(
        InstructionBuilder &builder,
        uint32_t base_ptr, uint32_t tile_idx,
        uint32_t val_id,
        uint32_t ptr_float_type_id,
        uint32_t float_type_id)
    {
      for (uint32_t i = 0; i < kTileSize; i++)
      {
        uint32_t idx_id = GetOrCreateIntConstant(tile_idx * kTileSize + i);
        auto *ptr = builder.AddAccessChain(ptr_float_type_id,
                                           base_ptr, {idx_id});
        auto *elem =
            builder.AddCompositeExtract(float_type_id, val_id, {i});
        builder.AddStore(ptr->result_id(), elem->result_id());
      }
    }

    // ===== 类型创建辅助 =====

    uint32_t CooperativeVecMatMulFallbackPass::EnsureVec4Type(
        uint32_t float_type_id)
    {
      for (auto &inst : context()->module()->types_values())
      {
        if (inst.opcode() == SpvOpTypeVector &&
            inst.GetSingleWordInOperand(0) == float_type_id &&
            inst.GetSingleWordInOperand(1) == kTileSize)
        {
          return inst.result_id();
        }
      }
      uint32_t id = context()->TakeNextId();
      context()->module()->AddType(MakeUnique<Instruction>(
          context(), SpvOpTypeVector, 0, id,
          {{SPV_OPERAND_TYPE_ID, {float_type_id}},
           {SPV_OPERAND_TYPE_LITERAL_INTEGER, {kTileSize}}}));
      return id;
    }

    uint32_t CooperativeVecMatMulFallbackPass::EnsurePointerType(
        uint32_t type_id, SpvStorageClass storage)
    {
      for (auto &inst : context()->module()->types_values())
      {
        if (inst.opcode() == SpvOpTypePointer &&
            inst.GetSingleWordInOperand(0) == static_cast<uint32_t>(storage) &&
            inst.GetSingleWordInOperand(1) == type_id)
        {
          return inst.result_id();
        }
      }
      uint32_t id = context()->TakeNextId();
      context()->module()->AddType(MakeUnique<Instruction>(
          context(), SpvOpTypePointer, 0, id,
          {{SPV_OPERAND_TYPE_STORAGE_CLASS, {static_cast<uint32_t>(storage)}},
           {SPV_OPERAND_TYPE_ID, {type_id}}}));
      return id;
    }

    // ===== 常量辅助 =====

    uint32_t CooperativeVecMatMulFallbackPass::GetOrCreateIntConstant(
        uint32_t value)
    {
      // 查找已有的 uint32 类型
      uint32_t uint_type_id = 0;
      for (auto &inst : context()->module()->types_values())
      {
        if (inst.opcode() == SpvOpTypeInt &&
            inst.GetSingleWordInOperand(0) == 32 &&
            inst.GetSingleWordInOperand(1) == 0)
        {
          uint_type_id = inst.result_id();
          break;
        }
      }
      if (uint_type_id == 0)
      {
        uint_type_id = context()->TakeNextId();
        context()->module()->AddType(MakeUnique<Instruction>(
            context(), SpvOpTypeInt, 0, uint_type_id,
            {{SPV_OPERAND_TYPE_LITERAL_INTEGER, {32}},
             {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0}}}));
      }

      // 查找已有的常量
      for (auto &inst : context()->module()->types_values())
      {
        if (inst.opcode() == SpvOpConstant &&
            inst.type_id() == uint_type_id &&
            inst.GetSingleWordInOperand(0) == value)
        {
          return inst.result_id();
        }
      }

      // 创建新常量
      uint32_t id = context()->TakeNextId();
      context()->module()->AddGlobalValue(MakeUnique<Instruction>(
          context(), SpvOpConstant, uint_type_id, id,
          {{SPV_OPERAND_TYPE_LITERAL_INTEGER, {value}}}));
      return id;
    }

    // ===== 指针类型辅助 =====

    // 从指针指令获取其 pointee 类型 id
    // 指针类型: OpTypePointer %storage_class %pointee_type_id
    // InOperand(0) = storage_class, InOperand(1) = pointee_type_id
    uint32_t CooperativeVecMatMulFallbackPass::GetPointeeTypeId(
        Instruction *ptr_inst)
    {
      uint32_t ptr_type_id = ptr_inst->type_id();
      auto *ptr_type_inst = get_def_use_mgr()->GetDef(ptr_type_id);
      return ptr_type_inst->GetSingleWordInOperand(1);
    }

  } // namespace opt
} // namespace spvtools
