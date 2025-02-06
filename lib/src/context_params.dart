// ignore_for_file: constant_identifier_names
part of '../llama.dart';

class ContextParams {
  // text context, 0 = from model
  int nCtx;

  // logical maximum batch size that can be submitted to llama_decode
  int nBatch;

  // physical maximum batch size
  int nUBatch;

  // max number of sequences (i.e. distinct states for recurrent models)
  int nSeqMax;

  // number of threads to use for generation
  int nThreads;

  // number of threads to use for batch processing
  int nThreadsBatch;

  // RoPE scaling type, from `enum llama_rope_scaling_type`
  RopeScalingType ropeScalingType;

  // whether to pool (sum) embedding results by sequence id
  PoolingType poolingType;

  // attention type to use for embeddings
  AttentionType attentionType;

  // RoPE base frequency, 0 = from model
  double ropeFrequencyBase;

  // RoPE frequency scaling factor, 0 = from model
  double ropeFrequencyScale;

  // YaRN extrapolation mix factor, negative = from model
  double yarnExtrapolationFactor;

  // YaRN magnitude scaling factor
  double yarnAttenuationFactor;

  // YaRN low correction dim
  double yarnBetaFast;

  // YaRN high correction dim
  double yarnBetaSlow;

  // YaRN original context size
  int yarnOriginalContext;

  // defragment the KV cache if holes/size > thold, < 0 disabled (default)
  double defragmentationThreshold;

  // data type for K cache
  GgmlType typeK;

  // data type for V cache
  GgmlType typeV;

  // if true, extract logits for each token
  bool logitsAll;

  // if true, extract embeddings (together with logits)
  bool embeddings;

  // whether to offload the KQV ops (including the KV cache) to GPU
  bool offloadKqv;

  // whether to use flash attention
  bool flashAttention;

  // whether to measure performance timings
  bool noPerformance;

  ContextParams({
    this.nCtx = 512,
    this.nBatch = 2048,
    this.nUBatch = 512,
    this.nSeqMax = 1,
    this.nThreads = 4,
    this.nThreadsBatch = 4,
    this.ropeScalingType = RopeScalingType.unspecified,
    this.poolingType = PoolingType.unspecified,
    this.attentionType = AttentionType.unspecified,
    this.ropeFrequencyBase = 0.0,
    this.ropeFrequencyScale = 0.0,
    this.yarnExtrapolationFactor = -1.0,
    this.yarnAttenuationFactor = 1.0,
    this.yarnBetaFast = 32.0,
    this.yarnBetaSlow = 1.0,
    this.yarnOriginalContext = 0,
    this.defragmentationThreshold = -1.0,
    this.typeK = GgmlType.f16,
    this.typeV = GgmlType.f16,
    this.logitsAll = false,
    this.embeddings = false,
    this.offloadKqv = true,
    this.flashAttention = false,
    this.noPerformance = true,
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
    final llama_context_params contextParams = calloc<llama_context_params>().ref;

    contextParams.n_ctx = nCtx;
    contextParams.n_batch = nBatch;
    contextParams.n_ubatch = nUBatch;
    contextParams.n_seq_max = nSeqMax;
    contextParams.n_threads = nThreads;
    contextParams.n_threads_batch = nThreadsBatch;
    contextParams.rope_scaling_type = ropeScalingType.index - 1;
    contextParams.pooling_type = poolingType.index - 1;
    contextParams.attention_type = attentionType.index - 1;
    contextParams.rope_freq_base = ropeFrequencyBase;
    contextParams.rope_freq_scale = ropeFrequencyScale;
    contextParams.yarn_ext_factor = yarnExtrapolationFactor;
    contextParams.yarn_attn_factor = yarnAttenuationFactor;
    contextParams.yarn_beta_fast = yarnBetaFast;
    contextParams.yarn_beta_slow = yarnBetaSlow;
    contextParams.yarn_orig_ctx = yarnOriginalContext;
    contextParams.defrag_thold = defragmentationThreshold;
    contextParams.type_k = typeK.index;
    contextParams.type_v = typeV.index;
    contextParams.logits_all = logitsAll;
    contextParams.embeddings = embeddings;
    contextParams.offload_kqv = offloadKqv;
    contextParams.flash_attn = flashAttention;
    contextParams.no_perf = noPerformance;

    return contextParams;
  }

  Map<String, dynamic> toMap() => {
    'nCtx': nCtx,
    'nBatch': nBatch,
    'nUBatch': nUBatch,
    'nSeqMax': nSeqMax,
    'nThreads': nThreads,
    'nThreadsBatch': nThreadsBatch,
    'ropeScalingType': ropeScalingType.name,
    'poolingType': poolingType.name,
    'attentionType': attentionType.name,
    'ropeFrequencyBase': ropeFrequencyBase,
    'ropeFrequencyScale': ropeFrequencyScale,
    'yarnExtrapolationFactor': yarnExtrapolationFactor,
    'yarnAttenuationFactor': yarnAttenuationFactor,
    'yarnBetaFast': yarnBetaFast,
    'yarnBetaSlow': yarnBetaSlow,
    'yarnOriginalContext': yarnOriginalContext,
    'defragmentationThreshold': defragmentationThreshold,
    'typeK': typeK.name,
    'typeV': typeV.name,
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