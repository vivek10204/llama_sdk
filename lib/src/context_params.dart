// ignore_for_file: constant_identifier_names
part of '../llama.dart';

class ContextParams {
  // text context, 0 = from model
  int? nCtx;

  // logical maximum batch size that can be submitted to llama_decode
  int? nBatch;

  // physical maximum batch size
  int? nUBatch;

  // max number of sequences (i.e. distinct states for recurrent models)
  int? nSeqMax;

  // number of threads to use for generation
  int? nThreads;

  // number of threads to use for batch processing
  int? nThreadsBatch;

  // RoPE scaling type, from `enum llama_rope_scaling_type`
  RopeScalingType? ropeScalingType;

  // whether to pool (sum) embedding results by sequence id
  PoolingType? poolingType;

  // attention type to use for embeddings
  AttentionType? attentionType;

  // RoPE base frequency, 0 = from model
  double? ropeFrequencyBase;

  // RoPE frequency scaling factor, 0 = from model
  double? ropeFrequencyScale;

  // YaRN extrapolation mix factor, negative = from model
  double? yarnExtrapolationFactor;

  // YaRN magnitude scaling factor
  double? yarnAttenuationFactor;

  // YaRN low correction dim
  double? yarnBetaFast;

  // YaRN high correction dim
  double? yarnBetaSlow;

  // YaRN original context size
  int? yarnOriginalContext;

  // defragment the KV cache if holes/size > thold, < 0 disabled (default)
  double? defragmentationThreshold;

  // data type for K cache
  GgmlType? typeK;

  // data type for V cache
  GgmlType? typeV;

  // if true, extract embeddings (together with logits)
  bool? embeddings;

  // whether to offload the KQV ops (including the KV cache) to GPU
  bool? offloadKqv;

  // whether to use flash attention
  bool? flashAttention;

  // whether to measure performance timings
  bool? noPerformance;

  ContextParams({
    this.nCtx,
    this.nBatch,
    this.nUBatch,
    this.nSeqMax,
    this.nThreads,
    this.nThreadsBatch,
    this.ropeScalingType,
    this.poolingType,
    this.attentionType,
    this.ropeFrequencyBase,
    this.ropeFrequencyScale,
    this.yarnExtrapolationFactor,
    this.yarnAttenuationFactor,
    this.yarnBetaFast,
    this.yarnBetaSlow,
    this.yarnOriginalContext,
    this.defragmentationThreshold,
    this.typeK,
    this.typeV,
    this.embeddings,
    this.offloadKqv,
    this.flashAttention,
    this.noPerformance,
  });

  /// Decodes a [ContextParams] instance from a list of bytes that was produced by
  /// [toBytes()]. Throws an exception if the bytes are malformed.
  factory ContextParams.fromBytes(List<int> bytes) {
    final buffer = Uint8List.fromList(bytes);
    final byteData = ByteData.sublistView(buffer);
    int offset = 0;
    // Read the header (first 4 bytes)
    final int header = byteData.getUint32(offset, Endian.little);
    offset += 4;

    final cp = ContextParams();

    if ((header & (1 << 0)) != 0) {
      cp.nCtx = byteData.getInt64(offset, Endian.little);
      offset += 8;
    }
    if ((header & (1 << 1)) != 0) {
      cp.nBatch = byteData.getInt64(offset, Endian.little);
      offset += 8;
    }
    if ((header & (1 << 2)) != 0) {
      cp.nUBatch = byteData.getInt64(offset, Endian.little);
      offset += 8;
    }
    if ((header & (1 << 3)) != 0) {
      cp.nSeqMax = byteData.getInt64(offset, Endian.little);
      offset += 8;
    }
    if ((header & (1 << 4)) != 0) {
      cp.nThreads = byteData.getInt64(offset, Endian.little);
      offset += 8;
    }
    if ((header & (1 << 5)) != 0) {
      cp.nThreadsBatch = byteData.getInt64(offset, Endian.little);
      offset += 8;
    }
    if ((header & (1 << 6)) != 0) {
      int index = byteData.getUint32(offset, Endian.little);
      cp.ropeScalingType = RopeScalingType.values[index];
      offset += 4;
    }
    if ((header & (1 << 7)) != 0) {
      int index = byteData.getUint32(offset, Endian.little);
      cp.poolingType = PoolingType.values[index];
      offset += 4;
    }
    if ((header & (1 << 8)) != 0) {
      int index = byteData.getUint32(offset, Endian.little);
      cp.attentionType = AttentionType.values[index];
      offset += 4;
    }
    if ((header & (1 << 9)) != 0) {
      cp.ropeFrequencyBase = byteData.getFloat64(offset, Endian.little);
      offset += 8;
    }
    if ((header & (1 << 10)) != 0) {
      cp.ropeFrequencyScale = byteData.getFloat64(offset, Endian.little);
      offset += 8;
    }
    if ((header & (1 << 11)) != 0) {
      cp.yarnExtrapolationFactor = byteData.getFloat64(offset, Endian.little);
      offset += 8;
    }
    if ((header & (1 << 12)) != 0) {
      cp.yarnAttenuationFactor = byteData.getFloat64(offset, Endian.little);
      offset += 8;
    }
    if ((header & (1 << 13)) != 0) {
      cp.yarnBetaFast = byteData.getFloat64(offset, Endian.little);
      offset += 8;
    }
    if ((header & (1 << 14)) != 0) {
      cp.yarnBetaSlow = byteData.getFloat64(offset, Endian.little);
      offset += 8;
    }
    if ((header & (1 << 15)) != 0) {
      cp.yarnOriginalContext = byteData.getInt64(offset, Endian.little);
      offset += 8;
    }
    if ((header & (1 << 16)) != 0) {
      cp.defragmentationThreshold = byteData.getFloat64(offset, Endian.little);
      offset += 8;
    }
    if ((header & (1 << 17)) != 0) {
      int index = byteData.getUint32(offset, Endian.little);
      cp.typeK = GgmlType.values[index];
      offset += 4;
    }
    if ((header & (1 << 18)) != 0) {
      int index = byteData.getUint32(offset, Endian.little);
      cp.typeV = GgmlType.values[index];
      offset += 4;
    }
    if ((header & (1 << 19)) != 0) {
      cp.embeddings = buffer[offset] != 0;
      offset += 1;
    }
    if ((header & (1 << 20)) != 0) {
      cp.offloadKqv = buffer[offset] != 0;
      offset += 1;
    }
    if ((header & (1 << 21)) != 0) {
      cp.flashAttention = buffer[offset] != 0;
      offset += 1;
    }
    if ((header & (1 << 22)) != 0) {
      cp.noPerformance = buffer[offset] != 0;
      offset += 1;
    }
    return cp;
  }

  llama_context_params toNative() {
    final llama_context_params contextParams = LlamaCppNative.lib.llama_context_default_params();

    if (nCtx != null) {
      contextParams.n_ctx = nCtx!;
    }

    if (nBatch != null) {
      contextParams.n_batch = nBatch!;
    }

    if (nUBatch != null) {
      contextParams.n_ubatch = nUBatch!;
    }

    if (nSeqMax != null) {
      contextParams.n_seq_max = nSeqMax!;
    }

    if (nThreads != null) {
      contextParams.n_threads = nThreads!;
    }

    if (nThreadsBatch != null) {
      contextParams.n_threads_batch = nThreadsBatch!;
    }

    if (ropeScalingType != null) {
      contextParams.rope_scaling_type = ropeScalingType!.index;
    }

    if (poolingType != null) {
      contextParams.pooling_type = poolingType!.index;
    }

    if (attentionType != null) {
      contextParams.attention_type = attentionType!.index;
    }

    if (ropeFrequencyBase != null) {
      contextParams.rope_freq_base = ropeFrequencyBase!;
    }

    if (ropeFrequencyScale != null) {
      contextParams.rope_freq_scale = ropeFrequencyScale!;
    }

    if (yarnExtrapolationFactor != null) {
      contextParams.yarn_ext_factor = yarnExtrapolationFactor!;
    }

    if (yarnAttenuationFactor != null) {
      contextParams.yarn_attn_factor = yarnAttenuationFactor!;
    }

    if (yarnBetaFast != null) {
      contextParams.yarn_beta_fast = yarnBetaFast!;
    }

    if (yarnBetaSlow != null) {
      contextParams.yarn_beta_slow = yarnBetaSlow!;
    }

    if (yarnOriginalContext != null) {
      contextParams.yarn_orig_ctx = yarnOriginalContext!;
    }

    if (defragmentationThreshold != null) {
      contextParams.defrag_thold = defragmentationThreshold!;
    }

    if (typeK != null) {
      contextParams.type_k = typeK!.index;
    }

    if (typeV != null) {
      contextParams.type_v = typeV!.index;
    }

    if (embeddings != null) {
      contextParams.embeddings = embeddings!;
    }

    if (offloadKqv != null) {
      contextParams.offload_kqv = offloadKqv!;
    }

    if (flashAttention != null) {
      contextParams.flash_attn = flashAttention!;
    }

    if (noPerformance != null) {
      contextParams.no_perf = noPerformance!;
    }

    return contextParams;
  }

  /// Encodes this instance into a list of bytes (List<int>) without losing
  /// double precision or null information.
  ///
  /// The encoding begins with a 4‑byte header (a bitmask indicating which of the
  /// 23 fields are non‑null) and then, in a fixed order, each present field is written:
  /// 
  /// - int fields as 64‑bit integers (8 bytes)
  /// - double fields as 64‑bit IEEE‑754 (8 bytes)
  /// - enum fields as 32‑bit unsigned integers (4 bytes)
  /// - booleans as 1 byte (0/1)
  List<int> toBytes() {
    // We define the order (indices 0..22) for our 23 fields:
    // 0: nCtx (int)
    // 1: nBatch (int)
    // 2: nUBatch (int)
    // 3: nSeqMax (int)
    // 4: nThreads (int)
    // 5: nThreadsBatch (int)
    // 6: ropeScalingType (enum)
    // 7: poolingType (enum)
    // 8: attentionType (enum)
    // 9: ropeFrequencyBase (double)
    // 10: ropeFrequencyScale (double)
    // 11: yarnExtrapolationFactor (double)
    // 12: yarnAttenuationFactor (double)
    // 13: yarnBetaFast (double)
    // 14: yarnBetaSlow (double)
    // 15: yarnOriginalContext (int)
    // 16: defragmentationThreshold (double)
    // 17: typeK (enum)
    // 18: typeV (enum)
    // 19: embeddings (bool)
    // 20: offloadKqv (bool)
    // 21: flashAttention (bool)
    // 22: noPerformance (bool)
    //
    // First, we calculate the total length needed.
    int length = 4; // 4 bytes for the header
    int header = 0;

    // For each field, if it is non-null, set the corresponding bit and add the size.
    if (nCtx != null)               { header |= (1 << 0);  length += 8; }
    if (nBatch != null)             { header |= (1 << 1);  length += 8; }
    if (nUBatch != null)            { header |= (1 << 2);  length += 8; }
    if (nSeqMax != null)            { header |= (1 << 3);  length += 8; }
    if (nThreads != null)           { header |= (1 << 4);  length += 8; }
    if (nThreadsBatch != null)      { header |= (1 << 5);  length += 8; }
    if (ropeScalingType != null)    { header |= (1 << 6);  length += 4; }
    if (poolingType != null)        { header |= (1 << 7);  length += 4; }
    if (attentionType != null)      { header |= (1 << 8);  length += 4; }
    if (ropeFrequencyBase != null)  { header |= (1 << 9);  length += 8; }
    if (ropeFrequencyScale != null) { header |= (1 << 10); length += 8; }
    if (yarnExtrapolationFactor != null) { header |= (1 << 11); length += 8; }
    if (yarnAttenuationFactor != null)   { header |= (1 << 12); length += 8; }
    if (yarnBetaFast != null)       { header |= (1 << 13); length += 8; }
    if (yarnBetaSlow != null)       { header |= (1 << 14); length += 8; }
    if (yarnOriginalContext != null){ header |= (1 << 15); length += 8; }
    if (defragmentationThreshold != null){ header |= (1 << 16); length += 8; }
    if (typeK != null)              { header |= (1 << 17); length += 4; }
    if (typeV != null)              { header |= (1 << 18); length += 4; }
    if (embeddings != null)         { header |= (1 << 19); length += 1; }
    if (offloadKqv != null)         { header |= (1 << 20); length += 1; }
    if (flashAttention != null)     { header |= (1 << 21); length += 1; }
    if (noPerformance != null)      { header |= (1 << 22); length += 1; }

    final buffer = Uint8List(length);
    final byteData = ByteData.sublistView(buffer);
    int offset = 0;
    // Write the 4‑byte header (the bitmask)
    byteData.setUint32(offset, header, Endian.little);
    offset += 4;

    // Now write each field if it was non-null.
    if (nCtx != null) {
      byteData.setInt64(offset, nCtx!, Endian.little);
      offset += 8;
    }
    if (nBatch != null) {
      byteData.setInt64(offset, nBatch!, Endian.little);
      offset += 8;
    }
    if (nUBatch != null) {
      byteData.setInt64(offset, nUBatch!, Endian.little);
      offset += 8;
    }
    if (nSeqMax != null) {
      byteData.setInt64(offset, nSeqMax!, Endian.little);
      offset += 8;
    }
    if (nThreads != null) {
      byteData.setInt64(offset, nThreads!, Endian.little);
      offset += 8;
    }
    if (nThreadsBatch != null) {
      byteData.setInt64(offset, nThreadsBatch!, Endian.little);
      offset += 8;
    }
    if (ropeScalingType != null) {
      byteData.setUint32(offset, ropeScalingType!.index, Endian.little);
      offset += 4;
    }
    if (poolingType != null) {
      byteData.setUint32(offset, poolingType!.index, Endian.little);
      offset += 4;
    }
    if (attentionType != null) {
      byteData.setUint32(offset, attentionType!.index, Endian.little);
      offset += 4;
    }
    if (ropeFrequencyBase != null) {
      byteData.setFloat64(offset, ropeFrequencyBase!, Endian.little);
      offset += 8;
    }
    if (ropeFrequencyScale != null) {
      byteData.setFloat64(offset, ropeFrequencyScale!, Endian.little);
      offset += 8;
    }
    if (yarnExtrapolationFactor != null) {
      byteData.setFloat64(offset, yarnExtrapolationFactor!, Endian.little);
      offset += 8;
    }
    if (yarnAttenuationFactor != null) {
      byteData.setFloat64(offset, yarnAttenuationFactor!, Endian.little);
      offset += 8;
    }
    if (yarnBetaFast != null) {
      byteData.setFloat64(offset, yarnBetaFast!, Endian.little);
      offset += 8;
    }
    if (yarnBetaSlow != null) {
      byteData.setFloat64(offset, yarnBetaSlow!, Endian.little);
      offset += 8;
    }
    if (yarnOriginalContext != null) {
      byteData.setInt64(offset, yarnOriginalContext!, Endian.little);
      offset += 8;
    }
    if (defragmentationThreshold != null) {
      byteData.setFloat64(offset, defragmentationThreshold!, Endian.little);
      offset += 8;
    }
    if (typeK != null) {
      byteData.setUint32(offset, typeK!.index, Endian.little);
      offset += 4;
    }
    if (typeV != null) {
      byteData.setUint32(offset, typeV!.index, Endian.little);
      offset += 4;
    }
    if (embeddings != null) {
      buffer[offset] = embeddings! ? 1 : 0;
      offset += 1;
    }
    if (offloadKqv != null) {
      buffer[offset] = offloadKqv! ? 1 : 0;
      offset += 1;
    }
    if (flashAttention != null) {
      buffer[offset] = flashAttention! ? 1 : 0;
      offset += 1;
    }
    if (noPerformance != null) {
      buffer[offset] = noPerformance! ? 1 : 0;
      offset += 1;
    }

    return buffer;
  }
}

enum RopeScalingType {
  unspecified,
  none,
  linear,
  yarn,
  longrope;
}

enum PoolingType {
  unspecified,
  none,
  mean,
  cls,
  last,
  rank;
}

enum AttentionType {
  unspecified,
  causal,
  nonCausal;
}

enum GgmlType {
  f32,
  f16,
  q4_0,
  q4_1,
  q4_2,
  q4_3,
  q5_0,
  q5_1,
  q8_0,
  q8_1,
  q2_k,
  q3_k,
  q4_k,
  q5_k,
  q6_k,
  q8_k,
  iq2_xxs,
  iq2_xs,
  iq3_xxs,
  iq1_s,
  iq4_nl,
  iq3_s,
  iq2_s,
  iq4_xs,
  i8,
  i16,
  i32,
  i64,
  f64,
  iq1_m,
  bf16,
  q4_0_4_4,
  q4_0_4_8,
  q4_0_8_8,
  tq1_0,
  tq2_0;
}