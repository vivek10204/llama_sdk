part of '../llama.dart';

class LlamaNative implements Llama {
  ffi.Pointer<llama_model> _model = ffi.nullptr;
  ffi.Pointer<llama_context> _context = ffi.nullptr;
  ffi.Pointer<llama_sampler> _sampler = ffi.nullptr;

  ModelParams _modelParams;
  ContextParams _contextParams; 
  SamplingParams _samplingParams;

  int _contextLength = 0;

  Completer? _completer;

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
    required ModelParams modelParams,
    ContextParams contextParams = const ContextParams(),
    SamplingParams samplingParams = const SamplingParams()
  }) : _modelParams = modelParams, 
       _contextParams = contextParams, 
       _samplingParams = samplingParams {
    Llama.lib.ggml_backend_load_all();
    Llama.lib.llama_backend_init();

    _initModel();
  }

  void _initModel() {
    final nativeModelParams = _modelParams.toNative();
    final nativeModelPath = _modelParams.path.toNativeUtf8().cast<ffi.Char>();
      
    if (_model != ffi.nullptr) {
      Llama.lib.llama_free_model(_model);
    }

    _model = Llama.lib.llama_load_model_from_file(
      nativeModelPath, 
      nativeModelParams
    );
    assert(_model != ffi.nullptr, 'Failed to load model');

    _initContext();
    _initSampler();
  }

  void _initContext() {
    final nativeContextParams = _contextParams.toNative();

    if (_context != ffi.nullptr) {
      Llama.lib.llama_free(_context);
    }

    _context = Llama.lib.llama_init_from_model(_model, nativeContextParams);
    assert(_context != ffi.nullptr, 'Failed to initialize context');
  }

  void _initSampler() {
    if (_sampler != ffi.nullptr) {
      Llama.lib.llama_sampler_free(_sampler);
    }

    final vocab = Llama.lib.llama_model_get_vocab(_model);
    _sampler = _samplingParams.toNative(vocab);
    assert(_sampler != ffi.nullptr, 'Failed to initialize sampler');
  }

  @override
  Future<String> prompt(List<ChatMessage> messages) async {
    assert(_model != ffi.nullptr, 'Model is not loaded');
    assert(_context != ffi.nullptr, 'Context is not initialized');
    assert(_sampler != ffi.nullptr, 'Sampler is not initialized');

    _completer = Completer();

    final nCtx = Llama.lib.llama_n_ctx(_context);

    ffi.Pointer<ffi.Char> formatted = calloc<ffi.Char>(nCtx);

    final template = Llama.lib.llama_model_chat_template(_model, ffi.nullptr);

    int newContextLength = Llama.lib.llama_chat_apply_template(
      template, 
      messages.toNative(), 
      messages.length, 
      true, 
      formatted, 
      nCtx
    );

    if (newContextLength > nCtx) {
      formatted = calloc<ffi.Char>(newContextLength);
      newContextLength = Llama.lib.llama_chat_apply_template(
        template, 
        messages.toNative(), 
        messages.length, 
        true, 
        formatted, 
        newContextLength
      );
    }

    if (newContextLength < 0) {
      throw Exception('Failed to apply template');
    }

    final prompt = formatted.cast<Utf8>().toDartString().substring(_contextLength, newContextLength);

    final vocab = Llama.lib.llama_model_get_vocab(_model);
    final isFirst = Llama.lib.llama_get_kv_cache_used_cells(_context) == 0;

    final nPromptTokens = -Llama.lib.llama_tokenize(vocab, prompt.toNativeUtf8().cast<ffi.Char>(), prompt.length, ffi.nullptr, 0, isFirst, true);
    ffi.Pointer<llama_token> promptTokens = calloc<llama_token>(nPromptTokens);

    if (Llama.lib.llama_tokenize(vocab, prompt.toNativeUtf8().cast<ffi.Char>(), prompt.length, promptTokens, nPromptTokens, isFirst, true) < 0) {
      throw Exception('Failed to tokenize');
    }

    llama_batch batch = Llama.lib.llama_batch_get_one(promptTokens, nPromptTokens);
    int newTokenId;

    String response = '';
    
    while (!_completer!.isCompleted) {
      final nCtx = Llama.lib.llama_n_ctx(_context);
      final nCtxUsed = Llama.lib.llama_get_kv_cache_used_cells(_context);

      if (nCtxUsed + batch.n_tokens > nCtx) {
        throw Exception('Context size exceeded');
      }

      if (Llama.lib.llama_decode(_context, batch) != 0) {
        throw Exception('Failed to decode');
      }

      newTokenId = Llama.lib.llama_sampler_sample(_sampler, _context, -1);

      // is it an end of generation?
      if (Llama.lib.llama_vocab_is_eog(vocab, newTokenId)) {
        break;
      }

      final buffer = calloc<ffi.Char>(256);
      final n = Llama.lib.llama_token_to_piece(vocab, newTokenId, buffer, 256, 0, true);
      if (n < 0) {
        throw Exception('Failed to convert token to piece');
      }

      response += buffer.cast<Utf8>().toDartString();

      final newTokenPointer = calloc<llama_token>(1);
      newTokenPointer.value = newTokenId;

      batch = Llama.lib.llama_batch_get_one(newTokenPointer, 1);
    }

    messages.add(ChatMessage(
      role: 'assistant',
      content: response
    ));

    _contextLength = Llama.lib.llama_chat_apply_template(
      template, 
      messages.toNative(), 
      messages.length, 
      false, 
      ffi.nullptr, 
      0
    );

    return messages.last.content;
  }

  @override
  void stop() {
    _completer?.complete();
  }

  void free() {
    Llama.lib.llama_free_model(_model);
    Llama.lib.llama_sampler_free(_sampler);
    Llama.lib.llama_free(_context);
  }
}