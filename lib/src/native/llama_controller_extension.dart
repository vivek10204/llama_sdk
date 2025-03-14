// ignore_for_file: constant_identifier_names

part of 'package:lcpp/lcpp.dart';

/// A class to manage the parameters for the llama model.
extension LlamaControllerExtension on LlamaController {
  /// Retrieves and initializes the model parameters for the llama model.
  ///
  /// This function initializes the model parameters using the default values
  /// provided by the llama library. It then updates the parameters based on
  /// the optional properties `vocabOnly`, `useMmap`, `useMlock`, and `checkTensors`
  /// if they are not null.
  ///
  /// Returns:
  ///   A `llama_model_params` object containing the initialized and updated
  ///   model parameters.
  llama_model_params getModelParams() {
    final llama_model_params modelParams = lib.llama_model_default_params();
    log("Model params initialized");

    if (vocabOnly != null) {
      modelParams.vocab_only = vocabOnly!;
    }

    if (useMmap != null) {
      modelParams.use_mmap = useMmap!;
    }

    if (useMlock != null) {
      modelParams.use_mlock = useMlock!;
    }

    if (checkTensors != null) {
      modelParams.check_tensors = checkTensors!;
    }

    return modelParams;
  }

  /// Returns a configured `llama_context_params` object based on the current instance's properties.
  ///
  /// This method initializes a `llama_context_params` object with default values and then overrides
  /// those values with the properties of the current instance if they are not null.
  ///
  /// The following properties are set:
  ///
  /// - `n_ctx`: The context size.
  /// - `n_batch`: The batch size.
  /// - `n_ubatch`: The unrolled batch size.
  /// - `n_seq_max`: The maximum sequence length.
  /// - `n_threads`: The number of threads.
  /// - `n_threads_batch`: The number of threads for batch processing.
  /// - `rope_scaling_type`: The type of rope scaling, adjusted by subtracting 1 from the enum index.
  /// - `pooling_type`: The type of pooling, adjusted by subtracting 1 from the enum index.
  /// - `attention_type`: The type of attention, adjusted by subtracting 1 from the enum index.
  /// - `rope_freq_base`: The base frequency for rope.
  /// - `rope_freq_scale`: The scaling factor for rope frequency.
  /// - `yarn_ext_factor`: The extrapolation factor for yarn.
  /// - `yarn_attn_factor`: The attenuation factor for yarn.
  /// - `yarn_beta_fast`: The fast beta value for yarn.
  /// - `yarn_beta_slow`: The slow beta value for yarn.
  /// - `yarn_orig_ctx`: The original context for yarn.
  /// - `defrag_thold`: The defragmentation threshold.
  /// - `type_k`: The type K, converted to a C int.
  /// - `type_v`: The type V, converted to a C int.
  /// - `embeddings`: The embeddings.
  /// - `offload_kqv`: The offload KQV flag.
  /// - `flash_attn`: The flash attention flag.
  /// - `no_perf`: The no performance flag.
  ///
  /// Returns:
  /// - A `llama_context_params` object with the configured properties.
  llama_context_params getContextParams() {
    final llama_context_params contextParams =
        lib.llama_context_default_params();

    contextParams.n_ctx = nCtx;

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

  /// Initializes and returns a pointer to a `llama_sampler` with the specified parameters.
  ///
  /// This method creates a sampler chain and adds various samplers to it based on the provided
  /// parameters. The samplers are added in the following order:
  ///
  /// 1. Greedy sampler (if `greedy` is true).
  /// 2. Infill sampler (if `_infill` is true, requires `vocab` to be non-null).
  /// 3. Distribution sampler (if `_seed` is not null).
  /// 4. Top-K sampler (if `_topK` is not null).
  /// 5. Top-P sampler (if both `_topP` and `_minKeepTopP` are not null).
  /// 6. Min-P sampler (if both `_minP` and `_minKeepMinP` are not null).
  /// 7. Typical sampler (if both `_typicalP` and `_minKeepTypicalP` are not null).
  /// 8. Temperature sampler (if `_temperature` is not null, with optional delta and exponent).
  /// 9. XTC sampler (if `_xtcP`, `_xtcT`, `_minKeepXtc`, and `_xtcSeed` are all not null).
  /// 10. Mirostat sampler (if `_mirostatNVocab`, `_mirostatSeed`, `_mirostatTau`, `_mirostatEta`, and `_mirostatM` are all not null).
  /// 11. Mirostat V2 sampler (if `_mirostatV2Seed`, `_mirostatV2Tau`, and `_mirostatV2Eta` are all not null).
  /// 12. Grammar sampler (if `_grammarStr` and `_grammarRoot` are not null, requires `vocab` to be non-null).
  /// 13. Penalties sampler (if `_penaltiesLastN`, `_penaltiesRepeat`, `_penaltiesFrequency`, and `_penaltiesPresent` are all not null).
  /// 14. Dry sampler (if `_drySamplerSequenceBreakers`, `_drySamplerNCtxTrain`, `_drySamplerMultiplier`, `_drySamplerDryBase`, and `_drySamplerAllowedLength` are all not null, requires `vocab` to be non-null).
  ///
  /// Parameters:
  /// - [vocab] (optional): A pointer to a `llama_vocab` required for certain samplers.
  ///
  /// Returns:
  /// - A pointer to the initialized `llama_sampler`.
  ///
  /// Throws:
  /// - `LlamaException` if `vocab` is required but not provided.
  ffi.Pointer<llama_sampler> getSampler([ffi.Pointer<llama_vocab>? vocab]) {
    final sampler =
        lib.llama_sampler_chain_init(lib.llama_sampler_chain_default_params());

    if (greedy) {
      lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_greedy());
    }

    if (infill) {
      assert(
          vocab != null, LlamaException('Vocabulary is required for infill'));
      lib.llama_sampler_chain_add(
          sampler, lib.llama_sampler_init_infill(vocab!));
    }

    if (seed != null) {
      lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_dist(seed!));
    }

    if (topK != null) {
      lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_top_k(topK!));
    }

    if (topP != null && minKeepTopP != null) {
      lib.llama_sampler_chain_add(
          sampler, lib.llama_sampler_init_top_p(topP!, minKeepTopP!));
    }

    if (minP != null && minKeepMinP != null) {
      lib.llama_sampler_chain_add(
          sampler, lib.llama_sampler_init_min_p(minP!, minKeepMinP!));
    }

    if (typicalP != null && minKeepTypicalP != null) {
      lib.llama_sampler_chain_add(
          sampler, lib.llama_sampler_init_typical(typicalP!, minKeepTypicalP!));
    }

    if (temperature != null) {
      if (temperatureDelta == null || temperatureExponent == null) {
        lib.llama_sampler_chain_add(
            sampler, lib.llama_sampler_init_temp(temperature!));
      } else {
        lib.llama_sampler_chain_add(
            sampler,
            lib.llama_sampler_init_temp_ext(
                temperature!, temperatureDelta!, temperatureExponent!));
      }
    }

    if (xtcP != null && xtcT != null && minKeepXtc != null && xtcSeed != null) {
      lib.llama_sampler_chain_add(sampler,
          lib.llama_sampler_init_xtc(xtcP!, xtcT!, minKeepXtc!, xtcSeed!));
    }

    if (mirostatNVocab != null &&
        mirostatSeed != null &&
        mirostatTau != null &&
        mirostatEta != null &&
        mirostatM != null) {
      lib.llama_sampler_chain_add(
          sampler,
          lib.llama_sampler_init_mirostat(mirostatNVocab!, mirostatSeed!,
              mirostatTau!, mirostatEta!, mirostatM!));
    }

    if (mirostatV2Seed != null &&
        mirostatV2Tau != null &&
        mirostatV2Eta != null) {
      lib.llama_sampler_chain_add(
          sampler,
          lib.llama_sampler_init_mirostat_v2(
              mirostatV2Seed!, mirostatV2Tau!, mirostatV2Eta!));
    }

    if (grammarStr != null && grammarRoot != null) {
      assert(
          vocab != null, LlamaException('Vocabulary is required for grammar'));
      lib.llama_sampler_chain_add(
          sampler,
          lib.llama_sampler_init_grammar(
              vocab!,
              grammarStr!.toNativeUtf8().cast<ffi.Char>(),
              grammarRoot!.toNativeUtf8().cast<ffi.Char>()));
    }

    if (penaltiesLastN != null &&
        penaltiesRepeat != null &&
        penaltiesFrequency != null &&
        penaltiesPresent != null) {
      lib.llama_sampler_chain_add(
          sampler,
          lib.llama_sampler_init_penalties(penaltiesLastN!, penaltiesRepeat!,
              penaltiesFrequency!, penaltiesPresent!));
    }

    if (drySamplerSequenceBreakers != null &&
        drySamplerNCtxTrain != null &&
        drySamplerMultiplier != null &&
        drySamplerDryBase != null &&
        drySamplerAllowedLength != null) {
      assert(vocab != null,
          LlamaException('Vocabulary is required for dry sampler'));
      final sequenceBreakers =
          calloc<ffi.Pointer<ffi.Char>>(drySamplerSequenceBreakers!.length);
      for (var i = 0; i < drySamplerSequenceBreakers!.length; i++) {
        sequenceBreakers[i] =
            drySamplerSequenceBreakers![i].toNativeUtf8().cast<ffi.Char>();
      }

      lib.llama_sampler_chain_add(
          sampler,
          lib.llama_sampler_init_dry(
              vocab!,
              drySamplerNCtxTrain!,
              drySamplerMultiplier!,
              drySamplerDryBase!,
              drySamplerAllowedLength!,
              drySamplerPenaltyLastN!,
              sequenceBreakers,
              drySamplerSequenceBreakers!.length));
    }

    return sampler;
  }
}
