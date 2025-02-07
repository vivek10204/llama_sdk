part of '../llama.dart';

class LlamaCppNative {
  ffi.Pointer<llama_model> _model = ffi.nullptr;

  String _modelPath;
  ModelParams _modelParams;
  ContextParams contextParams; 
  SamplingParams samplingParams;

  Completer? _completer;

  set modelPath(String modelPath) {
    _modelPath = modelPath;

    _initModel();
  }

  set modelParams(ModelParams modelParams) {
    _modelParams = modelParams;

    _initModel();
  }

  LlamaCppNative({
    required String modelPath,
    required ModelParams modelParams,
    required this.contextParams,
    required this.samplingParams
  }) : _modelPath = modelPath, _modelParams = modelParams {
    _initModel();
  }

  void _initModel() {
    final nativeModelParams = _modelParams.toNative();
    final nativeModelPath = _modelPath.toNativeUtf8().cast<ffi.Char>();
      
    if (_model != ffi.nullptr) {
      lib.llama_free_model(_model);
    }

    _model = lib.llama_load_model_from_file(
      nativeModelPath, 
      nativeModelParams
    );
  }

  Stream<String> prompt(List<ChatMessage> messages) {
    assert(_model != ffi.nullptr, 'Model is not loaded');

    _completer = Completer();

    final nCtx = contextParams.nCtx;

    ffi.Pointer<ffi.Char> formatted = calloc<ffi.Char>(nCtx);

    final template = lib.llama_model_chat_template(_model, ffi.nullptr);

    int contextLength = lib.llama_chat_apply_template(
      template, 
      messages.toNative(), 
      messages.length, 
      true, 
      formatted, 
      nCtx
    );

    if (contextLength > nCtx) {
      formatted = calloc<ffi.Char>(contextLength);
      contextLength = lib.llama_chat_apply_template(
        template, 
        messages.toNative(), 
        messages.length, 
        true, 
        formatted, 
        contextLength
      );
    }

    if (contextLength < 0) {
      throw Exception('Failed to apply template');
    }

    final prompt = formatted.cast<Utf8>().toDartString();

    return _generate(prompt);
  }

  Stream<String> _generate(String prompt) async* {
    final nativeContextParams = contextParams.toNative();

    final context = lib.llama_init_from_model(_model, nativeContextParams);
    assert(context != ffi.nullptr, 'Failed to initialize context');

    final vocab = lib.llama_model_get_vocab(_model);
    final sampler = samplingParams.toNative(vocab);
    assert(sampler != ffi.nullptr, 'Failed to initialize sampler');

    final isFirst = lib.llama_get_kv_cache_used_cells(context) == 0;

    final nPromptTokens = -lib.llama_tokenize(vocab, prompt.toNativeUtf8().cast<ffi.Char>(), prompt.length, ffi.nullptr, 0, isFirst, true);
    ffi.Pointer<llama_token> promptTokens = calloc<llama_token>(nPromptTokens);

    if (lib.llama_tokenize(vocab, prompt.toNativeUtf8().cast<ffi.Char>(), prompt.length, promptTokens, nPromptTokens, isFirst, true) < 0) {
      throw Exception('Failed to tokenize');
    }

    llama_batch batch = lib.llama_batch_get_one(promptTokens, nPromptTokens);
    int newTokenId;
    
    while (!_completer!.isCompleted) {
      final nCtx = lib.llama_n_ctx(context);
      final nCtxUsed = lib.llama_get_kv_cache_used_cells(context);

      if (nCtxUsed + batch.n_tokens > nCtx) {
        throw Exception('Context size exceeded');
      }

      if (lib.llama_decode(context, batch) != 0) {
        throw Exception('Failed to decode');
      }

      newTokenId = lib.llama_sampler_sample(sampler, context, -1);

      // is it an end of generation?
      if (lib.llama_vocab_is_eog(vocab, newTokenId)) {
        break;
      }

      final buffer = calloc<ffi.Char>(256);
      if (lib.llama_token_to_piece(vocab, newTokenId, buffer, 256, 0, true) < 0) {
        throw Exception('Failed to convert token to piece');
      }

      try {
        yield buffer.cast<Utf8>().toDartString();
      }
      catch (e) {
        throw FormatException('Failed to parse UTF-8 string from buffer', e);
      }
      finally {
        calloc.free(buffer);
      }

      final newTokenPointer = calloc<llama_token>(1);
      newTokenPointer.value = newTokenId;

      batch = lib.llama_batch_get_one(newTokenPointer, 1);
      calloc.free(newTokenPointer);
    }

    lib.llama_batch_free(batch);
    lib.llama_sampler_free(sampler);
    lib.llama_free(context);
    calloc.free(promptTokens);
  }

  void stop() {
    _completer?.complete();
  }

  void free() {
    lib.llama_free_model(_model);
  }
}