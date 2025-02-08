part of '../llama.dart';

class LlamaNative {
  ffi.Pointer<llama_model> _model = ffi.nullptr;
  ffi.Pointer<llama_context> _context = ffi.nullptr;
  ffi.Pointer<llama_sampler> _sampler = ffi.nullptr;

  String _modelPath;
  ModelParams _modelParams;
  ContextParams _contextParams; 
  SamplingParams _samplingParams;

  Completer? _completer;

  set modelPath(String modelPath) {
    _modelPath = modelPath;

    _initModel();
  }

  set modelParams(ModelParams modelParams) {
    _modelParams = modelParams;

    _initModel();
  }

  set contextParams(ContextParams contextParams) {
    _contextParams = contextParams;

    _initContext();
  }

  set samplingParams(SamplingParams samplingParams) {
    _samplingParams = samplingParams;

    _initSampler();
  }

  LlamaNative({
    required String modelPath,
    ModelParams modelParams = const ModelParams(),
    ContextParams contextParams = const ContextParams(),
    SamplingParams samplingParams = const SamplingParams()
  }) : _modelPath = modelPath, 
       _modelParams = modelParams, 
       _contextParams = contextParams, 
       _samplingParams = samplingParams {
    lib.ggml_backend_load_all();
    lib.llama_backend_init();

    _initModel();
    _initContext();
    _initSampler();
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
    assert(_model != ffi.nullptr, 'Failed to load model');
  }

  void _initContext() {
    final nativeContextParams = _contextParams.toNative();

    if (_context != ffi.nullptr) {
      lib.llama_free(_context);
    }

    _context = lib.llama_init_from_model(_model, nativeContextParams);
    assert(_context != ffi.nullptr, 'Failed to initialize context');
  }

  void _initSampler() {
    if (_sampler != ffi.nullptr) {
      lib.llama_sampler_free(_sampler);
    }

    final vocab = lib.llama_model_get_vocab(_model);
    _sampler = _samplingParams.toNative(vocab);
    assert(_sampler != ffi.nullptr, 'Failed to initialize sampler');
  }

  Stream<String> prompt(List<ChatMessage> messages) async* {
    assert(_model != ffi.nullptr, 'Model is not loaded');

    _completer = Completer();

    final nCtx = lib.llama_n_ctx(_context);

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
      throw LlamaException('Failed to apply template');
    }

    final prompt = formatted.cast<Utf8>().toDartString();

    final vocab = lib.llama_model_get_vocab(_model);

    final isFirst = lib.llama_get_kv_cache_used_cells(_context) == 0;

    final nPromptTokens = -lib.llama_tokenize(vocab, prompt.toNativeUtf8().cast<ffi.Char>(), prompt.length, ffi.nullptr, 0, isFirst, true);
    ffi.Pointer<llama_token> promptTokens = calloc<llama_token>(nPromptTokens);

    if (lib.llama_tokenize(vocab, prompt.toNativeUtf8().cast<ffi.Char>(), prompt.length, promptTokens, nPromptTokens, isFirst, true) < 0) {
      throw LlamaException('Failed to tokenize');
    }

    llama_batch batch = lib.llama_batch_get_one(promptTokens, nPromptTokens);
    int newTokenId;
    
    while (!_completer!.isCompleted) {
      final nCtxUsed = lib.llama_get_kv_cache_used_cells(_context);

      if (nCtxUsed + batch.n_tokens > nCtx) {
        throw LlamaException('Context size exceeded');
      }

      if (lib.llama_decode(_context, batch) != 0) {
        throw LlamaException('Failed to decode');
      }

      newTokenId = lib.llama_sampler_sample(_sampler, _context, -1);

      // is it an end of generation?
      if (lib.llama_vocab_is_eog(vocab, newTokenId)) {
        break;
      }

      final buffer = calloc<ffi.Char>(256);
      if (lib.llama_token_to_piece(vocab, newTokenId, buffer, 256, 0, true) < 0) {
        throw LlamaException('Failed to convert token to piece');
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
    calloc.free(promptTokens);
  }

  void stop() {
    _completer?.complete();
  }

  void free() {
    lib.llama_free_model(_model);
    lib.llama_sampler_free(_sampler);
    lib.llama_free(_context);
  }
}