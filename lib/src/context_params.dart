// ignore_for_file: constant_identifier_names
part of '../llama.dart';

class ContextParams {
  // text context, 0 = from model
  final int? nCtx;

  // logical maximum batch size that can be submitted to llama_decode
  final int? nBatch;

  // physical maximum batch size
  final int? nUBatch;

  // max number of sequences (i.e. distinct states for recurrent models)
  final int? nSeqMax;

  // number of threads to use for generation
  final int? nThreads;

  // number of threads to use for batch processing
  final int? nThreadsBatch;

  // RoPE scaling type, from `enum llama_rope_scaling_type`
  final RopeScalingType? ropeScalingType;

  // whether to pool (sum) embedding results by sequence id
  final PoolingType? poolingType;

  // attention type to use for embeddings
  final AttentionType? attentionType;

  // RoPE base frequency, 0 = from model
  final double? ropeFrequencyBase;

  // RoPE frequency scaling factor, 0 = from model
  final double? ropeFrequencyScale;

  // YaRN extrapolation mix factor, negative = from model
  final double? yarnExtrapolationFactor;

  // YaRN magnitude scaling factor
  final double? yarnAttenuationFactor;

  // YaRN low correction dim
  final double? yarnBetaFast;

  // YaRN high correction dim
  final double? yarnBetaSlow;

  // YaRN original context size
  final int? yarnOriginalContext;

  // defragment the KV cache if holes/size > thold, < 0 disabled (default)
  final double? defragmentationThreshold;

  // data type for K cache
  final GgmlType? typeK;

  // data type for V cache
  final GgmlType? typeV;

  // if true, extract embeddings (together with logits)
  final bool? embeddings;

  // whether to offload the KQV ops (including the KV cache) to GPU
  final bool? offloadKqv;

  // whether to use flash attention
  final bool? flashAttention;

  // whether to measure performance timings
  final bool? noPerformance;

  const ContextParams({
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

  factory ContextParams.fromMap(Map<String, dynamic> map) => ContextParams(
    nCtx: map['nCtx'],
    nBatch: map['nBatch'],
    nUBatch: map['nUBatch'],
    nSeqMax: map['nSeqMax'],
    nThreads: map['nThreads'],
    nThreadsBatch: map['nThreadsBatch'],
    ropeScalingType: RopeScalingType.fromString(map['ropeScalingType']),
    poolingType: PoolingType.fromString(map['poolingType']),
    attentionType: AttentionType.fromString(map['attentionType']),
    ropeFrequencyBase: map['ropeFrequencyBase'],
    ropeFrequencyScale: map['ropeFrequencyScale'],
    yarnExtrapolationFactor: map['yarnExtrapolationFactor'],
    yarnAttenuationFactor: map['yarnAttenuationFactor'],
    yarnBetaFast: map['yarnBetaFast'],
    yarnBetaSlow: map['yarnBetaSlow'],
    yarnOriginalContext: map['yarnOriginalContext'],
    defragmentationThreshold: map['defragmentationThreshold'],
    typeK: GgmlType.fromString(map['typeK']),
    typeV: GgmlType.fromString(map['typeV']),
    embeddings: map['embeddings'],
    offloadKqv: map['offloadKqv'],
    flashAttention: map['flashAttention'],
    noPerformance: map['noPerformance'],
  );

  factory ContextParams.fromJson(String source) => ContextParams.fromMap(jsonDecode(source));

  llama_context_params toNative() {
    final llama_context_params contextParams = Llama.lib.llama_context_default_params();

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
      // This enum starts at -1, so we need to subtract 1 from the index
      contextParams.rope_scaling_typeAsInt = ropeScalingType!.index - 1;
    }

    if (poolingType != null) {
      // This enum starts at -1, so we need to subtract 1 from the index
      contextParams.pooling_typeAsInt = poolingType!.index - 1;
    }

    if (attentionType != null) {
      // This enum starts at -1, so we need to subtract 1 from the index
      contextParams.attention_typeAsInt = attentionType!.index - 1;
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
      // It may seem redundant to multiply by 1, but it's necessary to convert to a C int
      contextParams.type_kAsInt = typeK!.index * 1;
    }

    if (typeV != null) {
      // It may seem redundant to multiply by 1, but it's necessary to convert to a C int
      contextParams.type_vAsInt = typeV!.index * 1;
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

  Map<String, dynamic> toMap() => {
    'nCtx': nCtx,
    'nBatch': nBatch,
    'nUBatch': nUBatch,
    'nSeqMax': nSeqMax,
    'nThreads': nThreads,
    'nThreadsBatch': nThreadsBatch,
    'ropeScalingType': ropeScalingType?.name,
    'poolingType': poolingType?.name,
    'attentionType': attentionType?.name,
    'ropeFrequencyBase': ropeFrequencyBase,
    'ropeFrequencyScale': ropeFrequencyScale,
    'yarnExtrapolationFactor': yarnExtrapolationFactor,
    'yarnAttenuationFactor': yarnAttenuationFactor,
    'yarnBetaFast': yarnBetaFast,
    'yarnBetaSlow': yarnBetaSlow,
    'yarnOriginalContext': yarnOriginalContext,
    'defragmentationThreshold': defragmentationThreshold,
    'typeK': typeK?.name,
    'typeV': typeV?.name,
    'embeddings': embeddings,
    'offloadKqv': offloadKqv,
    'flashAttention': flashAttention,
    'noPerformance': noPerformance,
  };

  String toJson() => jsonEncode(toMap());
}

enum RopeScalingType {
  unspecified,
  none,
  linear,
  yarn,
  longrope;

  static RopeScalingType fromString(String value) {
    switch (value) {
      case 'none':
        return RopeScalingType.none;
      case 'linear':
        return RopeScalingType.linear;
      case 'yarn':
        return RopeScalingType.yarn;
      case 'longrope':
        return RopeScalingType.longrope;
      default:
        return RopeScalingType.unspecified;
    }
  }
}

enum PoolingType {
  unspecified,
  none,
  mean,
  cls,
  last,
  rank;

  static PoolingType fromString(String value) {
    switch (value) {
      case 'none':
        return PoolingType.none;
      case 'mean':
        return PoolingType.mean;
      case 'cls':
        return PoolingType.cls;
      case 'last':
        return PoolingType.last;
      case 'rank':
        return PoolingType.rank;
      default:
        return PoolingType.unspecified;
    }
  }
}

enum AttentionType {
  unspecified,
  causal,
  nonCausal;

  static AttentionType fromString(String value) {
    switch (value) {
      case 'causal':
        return AttentionType.causal;
      case 'non-causal':
        return AttentionType.nonCausal;
      default:
        return AttentionType.unspecified;
    }
  }
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

  static GgmlType fromString(String value) {
    switch (value) {
      case 'f32':
        return GgmlType.f32;
      case 'f16':
        return GgmlType.f16;
      case 'q4_0':
        return GgmlType.q4_0;
      case 'q4_1':
        return GgmlType.q4_1;
      case 'q4_2':
        return GgmlType.q4_2;
      case 'q4_3':
        return GgmlType.q4_3;
      case 'q5_0':
        return GgmlType.q5_0;
      case 'q5_1':
        return GgmlType.q5_1;
      case 'q8_0':
        return GgmlType.q8_0;
      case 'q8_1':
        return GgmlType.q8_1;
      case 'q2_k':
        return GgmlType.q2_k;
      case 'q3_k':
        return GgmlType.q3_k;
      case 'q4_k':
        return GgmlType.q4_k;
      case 'q5_k':
        return GgmlType.q5_k;
      case 'q6_k':
        return GgmlType.q6_k;
      case 'q8_k':
        return GgmlType.q8_k;
      case 'iq2_xxs':
        return GgmlType.iq2_xxs;
      case 'iq2_xs':
        return GgmlType.iq2_xs;
      case 'iq3_xxs':
        return GgmlType.iq3_xxs;
      case 'iq1_s':
        return GgmlType.iq1_s;
      case 'iq4_nl':
        return GgmlType.iq4_nl;
      case 'iq3_s':
        return GgmlType.iq3_s;
      case 'iq2_s':
        return GgmlType.iq2_s;
      case 'iq4_xs':
        return GgmlType.iq4_xs;
      case 'i8':
        return GgmlType.i8;
      case 'i16':
        return GgmlType.i16;
      case 'i32':
        return GgmlType.i32;
      case 'i64':
        return GgmlType.i64;
      case 'f64':
        return GgmlType.f64;
      case 'iq1_m':
        return GgmlType.iq1_m;
      case 'bf16':
        return GgmlType.bf16;
      case 'q4_0_4_4':
        return GgmlType.q4_0_4_4;
      case 'q4_0_4_8':
        return GgmlType.q4_0_4_8;
      case 'q4_0_8_8':
        return GgmlType.q4_0_8_8;
      case 'tq1_0':
        return GgmlType.tq1_0;
      case 'tq2_0':
        return GgmlType.tq2_0;
      default:
        return GgmlType.f32;
    }
  }
}