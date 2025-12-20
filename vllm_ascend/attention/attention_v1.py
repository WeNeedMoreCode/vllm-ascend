#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Type

import torch
import torch_npu
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils import direct_register_custom_op
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_ascend.ops.attention import vanilla_chunked_prefill
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ, aligned_16, is_310p,
                               nd_to_nz_2d, nd_to_nz_spec)
from vllm_ascend.worker.npu_input_batch import InputBatch


class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND"

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionBackendImpl"]:
        return AscendAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["AscendMetadata"]:
        return AscendMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if is_310p():
            return (2, num_blocks, num_kv_heads * head_size // 16, block_size,
                    16)
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_bsh_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads * head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: List[torch.Tensor],
        dst_kv_cache: List[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(
            dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(
            dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendMetadata:

    # **************************** Basic Properties ****************************
    attn_mask: Optional[torch.Tensor] = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    seq_lens: torch.Tensor = None

    query_start_loc: torch.Tensor = None
    query_lens: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: Optional[int] = None

    # ********************** KV Cache Related Properties ***********************
    # Block addresses per sequence (Seq id -> list of physical block).
    # (batch_size, max_blocks_per_seq)
    block_tables: torch.Tensor = None

    # The indices of the token slots that input tokens will be stored into.
    # E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the
    # three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0,
    # and 1st slot in block 1, respectively.
    # (num_tokens,)
    slot_mapping: torch.Tensor = None

    enable_dbo_across_dp: bool = False
    is_only_prefill: bool = False


class AscendAttentionMetadataBuilder:

    def __init__(self, runner):
        self.runner = runner

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(self,
              num_reqs,
              num_actual_tokens,
              max_query_len,
              enable_dbo_across_dp: bool = False,
              is_only_prefill: bool = False):

        block_table = self.runner.input_batch.block_table[0].get_device_tensor(
        )
        block_table[:num_reqs, :self.runner.max_num_blocks_per_req] = (
            block_table[:num_reqs])

        query_lens = self.runner.query_lens
        seq_lens = self.runner.seq_lens_cpu[:num_reqs]
        slot_mapping = self.runner.slot_mapping_cpu[:num_actual_tokens].to(
            self.runner.device, non_blocking=True)
        attn_mask = self.runner.attn_mask
        attn_state = self.runner.attn_state
        query_start_loc_cpu = self.runner.query_start_loc_cpu[:num_reqs + 1]
        query_start_loc = query_start_loc_cpu.to(self.runner.device,
                                                 non_blocking=True)

        if is_310p():
            if attn_state == AscendAttentionState.PrefillNoCache:
                mask_nz = nd_to_nz_2d(attn_mask)
                attn_mask = torch_npu.npu_format_cast(mask_nz.contiguous(),
                                                      ACL_FORMAT_FRACTAL_NZ)
            elif attn_state == AscendAttentionState.ChunkedPrefill:
                mask_nz = nd_to_nz_spec(attn_mask)
                attn_mask = torch_npu.npu_format_cast(mask_nz.contiguous(),
                                                      ACL_FORMAT_FRACTAL_NZ)

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            query_lens=query_lens,
            seq_lens=seq_lens,
            max_query_len=max_query_len,
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            enable_dbo_across_dp=enable_dbo_across_dp,
            is_only_prefill=is_only_prefill)
        return attn_metadata


class AscendAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes,
                                        dtype=torch.float32,
                                        device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        trace_flag: bool = True,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            kv_cache: shape = [key_cache, value_cache]
                      key_cache = [num_blocks, block_size,
                                   num_kv_heads, head_size]
                      value_cache = [num_blocks, block_size,
                                     num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size * seq_len, num_heads, head_size]
        """
        num_tokens = query.shape[0]
        use_kv_cache_int8 = len(
            kv_cache) > 0 and kv_cache[0].dtype == torch.int8
        if output is None:
            output = torch.empty(num_tokens,
                                 self.num_heads,
                                 self.head_size,
                                 dtype=query.dtype,
                                 device=query.device)
        ori_output = output
        if trace_flag:
            torch.ops.vllm.unified_ascend_attention_with_output(
                query=query,
                key=key,
                value=value,
                output=output,
                layer_name=layer.layer_name)

        elif hasattr(layer, 'quant_method') and use_kv_cache_int8:
            output = layer.quant_method.apply(layer, query, key, value,
                                              kv_cache, attn_metadata,
                                              self.attn_type, self.scale,
                                              output)

        else:
            if attn_metadata is None:
                return output.view(num_tokens, self.hidden_size)

            # DEBUG: 输出层名称用于对应新老版本的DEBUG信息
            layer_name = getattr(layer, 'layer_name', 'unknown_layer')
            print(f"[PRECISION DEBUG LAYER] ===== ENTERING LAYER: {layer_name} =====")

            # DEBUG: 检查第0层和第1层的投影权重
            if "layers.0." in layer_name or "layers.1." in layer_name:
                print(f"[PRECISION DEBUG OLD WEIGHTS] ANALYZING WEIGHTS FOR {layer_name}:")

                # 尝试获取投影权重 (假设layer有o_proj, q_proj, k_proj, v_proj等属性)
                weight_info = {}

                # 检查各种可能的权重属性
                for attr_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if hasattr(layer, attr_name):
                        proj_layer = getattr(layer, attr_name)
                        if hasattr(proj_layer, 'weight') and proj_layer.weight is not None:
                            weight = proj_layer.weight
                            weight_info[f'{attr_name}_weight'] = {
                                'shape': weight.shape,
                                'dtype': weight.dtype,
                                'mean': weight.float().mean().item(),
                                'std': weight.float().std().item(),
                                'min': weight.float().min().item(),
                                'max': weight.float().max().item(),
                                'device': weight.device
                            }

                            # 检查bias
                            if hasattr(proj_layer, 'bias') and proj_layer.bias is not None:
                                bias = proj_layer.bias
                                weight_info[f'{attr_name}_bias'] = {
                                    'shape': bias.shape,
                                    'dtype': bias.dtype,
                                    'mean': bias.float().mean().item(),
                                    'std': bias.float().std().item(),
                                    'min': bias.float().min().item(),
                                    'max': bias.float().max().item(),
                                    'device': bias.device
                                }

                # 输出权重信息
                for weight_key, weight_data in weight_info.items():
                    print(f"[PRECISION DEBUG OLD WEIGHTS]   {weight_key}:")
                    print(f"[PRECISION DEBUG OLD WEIGHTS]     shape={weight_data['shape']}")
                    print(f"[PRECISION DEBUG OLD WEIGHTS]     dtype={weight_data['dtype']}")
                    print(f"[PRECISION DEBUG OLD WEIGHTS]     device={weight_data['device']}")
                    print(f"[PRECISION DEBUG OLD WEIGHTS]     mean={weight_data['mean']:.8f}")
                    print(f"[PRECISION DEBUG OLD WEIGHTS]     std={weight_data['std']:.8f}")
                    print(f"[PRECISION DEBUG OLD WEIGHTS]     range=[{weight_data['min']:.8f}, {weight_data['max']:.8f}]")

            # DEBUG: 精度检查 - 老版本attention入口处的原始输入
            print(f"[PRECISION DEBUG OLD ENTRY] OLD VERSION:")
            print(f"[PRECISION DEBUG OLD ENTRY]   query_orig: shape={query.shape}; mean={query.float().mean().item():.6f}; std={query.float().std().item():.6f}")
            print(f"[PRECISION DEBUG OLD ENTRY]   key_orig: shape={key.shape}; mean={key.float().mean().item():.6f}; std={key.float().std().item():.6f}")
            print(f"[PRECISION DEBUG OLD ENTRY]   value_orig: shape={value.shape}; mean={value.float().mean().item():.6f}; std={value.float().std().item():.6f}")
            print(f"[PRECISION DEBUG OLD ENTRY]   attn_state: {attn_metadata.attn_state}")

            num_actual_tokens = attn_metadata.num_actual_tokens
            assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
            attn_type = self.attn_type
            if attn_type != AttentionType.DECODER:
                raise NotImplementedError("Encoder self-attention and "
                                          "encoder/decoder cross-attention "
                                          "are not implemented for "
                                          "PallasAttentionBackendImpl")
            # View q k v to BSH.
            query = query.view(-1, self.num_heads, self.head_size)
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
            # TODO: Remove this contiguous in the future.
            value = value.contiguous()

            if len(kv_cache) > 1:
                if self.key_cache is None:
                    self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
                slots = attn_metadata.slot_mapping

                # DEBUG: 精度检查 - 老版本reshape_and_cache前后的key/value变化
                print(f"[PRECISION DEBUG OLD BEFORE RESHAPE]:")
                print(f"[PRECISION DEBUG OLD BEFORE RESHAPE]   key_before: shape={key.shape}; mean={key.float().mean().item():.6f}; std={key.float().std().item():.6f}")
                print(f"[PRECISION DEBUG OLD BEFORE RESHAPE]   value_before: shape={value.shape}; mean={value.float().mean().item():.6f}; std={value.float().std().item():.6f}")
                print(f"[PRECISION DEBUG OLD BEFORE RESHAPE]   num_actual_tokens: {num_actual_tokens}")

                torch_npu._npu_reshape_and_cache(
                    key=key[:num_actual_tokens],
                    value=value[:num_actual_tokens],
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    slot_indices=slots)

                # DEBUG: 精度检查 - 老版本reshape_and_cache后key/value是否被修改
                print(f"[PRECISION DEBUG OLD AFTER RESHAPE]:")
                print(f"[PRECISION DEBUG OLD AFTER RESHAPE]   key_after: shape={key.shape}; mean={key.float().mean().item():.6f}; std={key.float().std().item():.6f}")
                print(f"[PRECISION DEBUG OLD AFTER RESHAPE]   value_after: shape={value.shape}; mean={value.float().mean().item():.6f}; std={value.float().std().item():.6f}")

            # V0-Style scheduler situation.
            if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
                print(f"[PRECISION DEBUG OLD PATH] Taking PrefillNoCache path (equivalent to NEW VERSION FALLBACK)")
                assert attn_metadata is not None
                assert attn_metadata.attn_mask is not None
                mask = attn_metadata.attn_mask
                if is_310p():
                    # DEBUG: 精度检查 - 老版本attention输入
                    print(f"[PRECISION DEBUG OLD VERSION ENTRY] OLD VERSION:")
                    print(f"[PRECISION DEBUG OLD VERSION ENTRY]   query_orig: shape={query.shape}; mean={query.float().mean().item():.6f}; std={query.float().std().item():.6f}")
                    print(f"[PRECISION DEBUG OLD VERSION ENTRY]   key_orig: shape={key.shape}; mean={key.float().mean().item():.6f}; std={key.float().std().item():.6f}")
                    print(f"[PRECISION DEBUG OLD VERSION ENTRY]   value_orig: shape={value.shape}; mean={value.float().mean().item():.6f}; std={value.float().std().item():.6f}")

                    # DEBUG: aligned_16前后的精度检查
                    print(f"[PRECISION DEBUG OLD VERSION ALIGNED_16] Before aligned_16:")
                    print(f"[PRECISION DEBUG OLD VERSION ALIGNED_16]   query_orig: shape={query.shape}; mean={query.float().mean().item():.6f}")
                    print(f"[PRECISION DEBUG OLD VERSION ALIGNED_16]   key_orig: shape={key.shape}; mean={key.float().mean().item():.6f}")
                    print(f"[PRECISION DEBUG OLD VERSION ALIGNED_16]   value_orig: shape={value.shape}; mean={value.float().mean().item():.6f}")

                    # align q k v output tensors
                    query = aligned_16(query)
                    key = aligned_16(key)
                    value = aligned_16(value)
                    output = aligned_16(output)

                    print(f"[PRECISION DEBUG OLD VERSION ALIGNED_16] After aligned_16:")
                    print(f"[PRECISION DEBUG OLD VERSION ALIGNED_16]   query_aligned: shape={query.shape}; mean={query.float().mean().item():.6f}")
                    print(f"[PRECISION DEBUG OLD VERSION ALIGNED_16]   key_aligned: shape={key.shape}; mean={key.float().mean().item():.6f}")
                    print(f"[PRECISION DEBUG OLD VERSION ALIGNED_16]   value_aligned: shape={value.shape}; mean={value.float().mean().item():.6f}")

                    # do reformat in case of broadcasted tensors
                    mask = mask.repeat(attn_metadata.seq_lens.size(0), 1, 1, 1)
                    mask = torch_npu.npu_format_cast(mask.contiguous(),
                                                     ACL_FORMAT_FRACTAL_NZ)

                torch_npu._npu_flash_attention(query=query,
                                               key=key,
                                               value=value,
                                               mask=mask,
                                               seq_len=attn_metadata.seq_lens,
                                               scale_value=self.scale,
                                               num_heads=self.num_heads,
                                               num_kv_heads=self.num_kv_heads,
                                               out=output)

                # DEBUG: 精度检查 - 老版本attention输出
                print(f"[PRECISION DEBUG OLD VERSION ATTENTION] After torch_npu._npu_flash_attention:")
                output_full_shape = output.shape
                print(f"[PRECISION DEBUG OLD VERSION ATTENTION]   output_full: shape={output_full_shape}; mean={output.float().mean().item():.6f}; std={output.float().std().item():.6f}")

                output = output[:num_tokens, :, :]
                print(f"[PRECISION DEBUG OLD VERSION ATTENTION]   final_output: shape={output.shape}; mean={output.float().mean().item():.6f}; std={output.float().std().item():.6f}")
            elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
                print(f"[PRECISION DEBUG OLD PATH] Taking PrefillCacheHit path")
                # DEBUG: 精度检查 - 老版本PrefillCacheHit路径的输入
                print(f"[PRECISION DEBUG OLD PREFILL_CACHE_HIT ENTRY] OLD VERSION:")
                print(f"[PRECISION DEBUG OLD PREFILL_CACHE_HIT ENTRY]   query: shape={query.shape}; mean={query.float().mean().item():.6f}; std={query.float().std().item():.6f}")

                assert attn_metadata is not None
                assert attn_metadata.attn_mask is not None
                compress_mask = attn_metadata.attn_mask
                batch_size = attn_metadata.query_lens.shape[0]
                block_table = attn_metadata.block_tables[:batch_size, :]
                torch_npu._npu_flash_attention_qlens(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    block_table=block_table,
                    mask=compress_mask,
                    seq_len=attn_metadata.query_lens,
                    context_lens=attn_metadata.seq_lens,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    out=output)

                # DEBUG: 精度检查 - 老版本PrefillCacheHit路径的输出
                print(f"[PRECISION DEBUG OLD PREFILL_CACHE_HIT EXIT] OLD VERSION:")
                print(f"[PRECISION DEBUG OLD PREFILL_CACHE_HIT EXIT]   output: shape={output.shape}; mean={output.float().mean().item():.6f}; std={output.float().std().item():.6f}")
            elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                print(f"[PRECISION DEBUG OLD PATH] Taking DecodeOnly path")
                # DEBUG: 精度检查 - 老版本DecodeOnly路径的输入
                print(f"[PRECISION DEBUG OLD DECODE_ONLY ENTRY] OLD VERSION:")
                print(f"[PRECISION DEBUG OLD DECODE_ONLY ENTRY]   query: shape={query.shape}; mean={query.float().mean().item():.6f}; std={query.float().std().item():.6f}")

                if is_310p():
                    # # seq_lens_tensor needs to be transferred to the device for 310P
                    attn_metadata.seq_lens = \
                        attn_metadata.seq_lens.to(device=query.device)
                torch_npu._npu_paged_attention(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output)

                # DEBUG: 精度检查 - 老版本DecodeOnly路径的输出
                print(f"[PRECISION DEBUG OLD DECODE_ONLY EXIT] OLD VERSION:")
                print(f"[PRECISION DEBUG OLD DECODE_ONLY EXIT]   output: shape={output.shape}; mean={output.float().mean().item():.6f}; std={output.float().std().item():.6f}")
            # Normal V1 situation.
            else:
                # DEBUG: 处理其他attn_state，包括ChunkedPrefill
                print(f"[PRECISION DEBUG OLD PATH] Taking other path (likely ChunkedPrefill), head_size={self.head_size}")
                # DEBUG: 精度检查 - 老版本其他路径的输入
                print(f"[PRECISION DEBUG OLD OTHER ENTRY] OLD VERSION:")
                print(f"[PRECISION DEBUG OLD OTHER ENTRY]   query: shape={query.shape}; mean={query.float().mean().item():.6f}; std={query.float().std().item():.6f}")

                # use chunked prefill for head size 192 scenario, like deepseek
                # paged_attention_splitfuse maybe crash at such scenario
                # TODO: vanilla path will be removed after the kernel support
                # head_size 192 scenario
                if self.head_size == 192:
                    cu_seqlen_q = [0] + attn_metadata.query_lens.tolist()
                    cu_seqlen_k = [0] + attn_metadata.seq_lens.tolist()
                    cu_seqlen_q = torch.tensor(cu_seqlen_q,
                                               device=query.device)
                    cu_seqlen_k = torch.tensor(cu_seqlen_k,
                                               device=query.device)
                    cu_seqlen_q = torch.cumsum(cu_seqlen_q, dim=0)
                    cu_seqlen_k = torch.cumsum(cu_seqlen_k, dim=0)
                    max_seqlen_q = torch.max(attn_metadata.query_lens)
                    max_seqlen_k = torch.max(attn_metadata.seq_lens)
                    vanilla_chunked_prefill(output, query, self.key_cache,
                                            self.value_cache,
                                            attn_metadata.block_tables,
                                            cu_seqlen_q, cu_seqlen_k,
                                            max_seqlen_q, max_seqlen_k,
                                            self.scale, None, True)
                else:
                    # use paged attention
                    assert attn_metadata is not None
                    assert attn_metadata.attn_mask is not None
                    if is_310p():
                        # do reformat in case of broadcasted tensors
                        attn_metadata.attn_mask = \
                            torch_npu.npu_format_cast(attn_metadata.attn_mask.contiguous(), ACL_FORMAT_FRACTAL_NZ)
                        attn_metadata.seq_lens = \
                            attn_metadata.seq_lens.to(device=query.device)
                    torch_npu._npu_paged_attention_splitfuse(
                        query=query,
                        key_cache=self.key_cache,
                        value_cache=self.value_cache,
                        mask=attn_metadata.attn_mask,
                        block_table=attn_metadata.block_tables,
                        seq_len=attn_metadata.query_lens,
                        context_lens=attn_metadata.seq_lens,
                        num_kv_heads=self.num_kv_heads,
                        num_heads=self.num_heads,
                        scale_value=self.scale,
                        out=output)

        # to make in-place change to the output tensor
        if hasattr(layer, 'quant_method') and use_kv_cache_int8:
            output = output.view(num_tokens, self.num_heads, self.head_size)
        ori_output[:, :, :] = output[:num_tokens, :, :]

        # DEBUG: 精度检查 - 老版本attention的最终输出
        print(f"[PRECISION DEBUG OLD FINAL EXIT] OLD VERSION:")
        print(f"[PRECISION DEBUG OLD FINAL EXIT]   final_output: shape={output.shape}; mean={output.float().mean().item():.6f}; std={output.float().std().item():.6f}")

        return output.view(num_tokens, self.hidden_size)


def unified_ascend_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    self.impl.forward(self,
                      query,
                      key,
                      value,
                      kv_cache,
                      attn_metadata,
                      output,
                      trace_flag=False)
    return


def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_ascend_attention_with_output",
    op_func=unified_ascend_attention_with_output,
    mutates_args=["output"],
    fake_impl=unified_attention_with_output_fake,
    dispatch_key="PrivateUse1",
)
