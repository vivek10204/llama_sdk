part of '../lcpp.dart';

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
    assert(_model != ffi.nullptr, 'Model is not loaded');

    final nativeContextParams = _contextParams.toNative();

    if (_context != ffi.nullptr) {
      Llama.lib.llama_free(_context);
    }

    _context = Llama.lib.llama_init_from_model(_model, nativeContextParams);
    assert(_context != ffi.nullptr, 'Failed to initialize context');
  }

  void _initSampler() {
    assert(_model != ffi.nullptr, 'Model is not loaded');

    if (_sampler != ffi.nullptr) {
      Llama.lib.llama_sampler_free(_sampler);
    }

    final vocab = Llama.lib.llama_model_get_vocab(_model);
    _sampler = _samplingParams.toNative(vocab);
    assert(_sampler != ffi.nullptr, 'Failed to initialize sampler');
  }

  @override
  Stream<String> prompt(List<ChatMessage> messages) async* {
    final messagesCopy = messages.copy();

    assert(_model != ffi.nullptr, 'Model is not loaded');
    assert(_context != ffi.nullptr, 'Context is not initialized');
    assert(_sampler != ffi.nullptr, 'Sampler is not initialized');

    _completer = Completer();

    final nCtx = Llama.lib.llama_n_ctx(_context);

    ffi.Pointer<ffi.Char> formatted = calloc<ffi.Char>(nCtx);

    final template = Llama.lib.llama_model_chat_template(_model, ffi.nullptr);

    ffi.Pointer<llama_chat_message> messagesPtr = messagesCopy.toNative();

    int newContextLength = Llama.lib.llama_chat_apply_template(
      template, 
      messagesPtr, 
      messagesCopy.length, 
      true, 
      formatted, 
      nCtx
    );

    if (newContextLength > nCtx) {
      calloc.free(formatted);
      formatted = calloc<ffi.Char>(newContextLength);
      newContextLength = Llama.lib.llama_chat_apply_template(
        template, 
        messagesPtr, 
        messagesCopy.length, 
        true, 
        formatted, 
        newContextLength
      );
    }

    messagesPtr.free(messagesCopy.length);

    if (newContextLength < 0) {
      throw Exception('Failed to apply template');
    }

    final prompt = formatted.cast<Utf8>().toDartString().substring(_contextLength);
    calloc.free(formatted);

    final vocab = Llama.lib.llama_model_get_vocab(_model);
    final isFirst = Llama.lib.llama_get_kv_cache_used_cells(_context) == 0;

    final promptPtr = prompt.toNativeUtf8().cast<ffi.Char>();

    final nPromptTokens = -Llama.lib.llama_tokenize(vocab, promptPtr, prompt.length, ffi.nullptr, 0, isFirst, true);
    ffi.Pointer<llama_token> promptTokens = calloc<llama_token>(nPromptTokens);

    if (Llama.lib.llama_tokenize(vocab, promptPtr, prompt.length, promptTokens, nPromptTokens, isFirst, true) < 0) {
      throw Exception('Failed to tokenize');
    }

    calloc.free(promptPtr);

    llama_batch batch = Llama.lib.llama_batch_get_one(promptTokens, nPromptTokens);
    ffi.Pointer<llama_token> newTokenId = calloc<llama_token>(1);

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

      newTokenId.value = Llama.lib.llama_sampler_sample(_sampler, _context, -1);

      // is it an end of generation?
      if (Llama.lib.llama_vocab_is_eog(vocab, newTokenId.value)) {
        break;
      }

      final buffer = calloc<ffi.Char>(256);
      final n = Llama.lib.llama_token_to_piece(vocab, newTokenId.value, buffer, 256, 0, true);
      if (n < 0) {
        throw Exception('Failed to convert token to piece');
      }

      final piece = buffer.cast<Utf8>().toDartString();
      calloc.free(buffer);
      response += piece;
      yield piece;

      batch = Llama.lib.llama_batch_get_one(newTokenId, 1);
    }

    messagesCopy.add(ChatMessage(
      role: 'assistant',
      content: response
    ));

    messagesPtr = messagesCopy.toNative();

    _contextLength = Llama.lib.llama_chat_apply_template(
      template, 
      messagesPtr, 
      messagesCopy.length, 
      false, 
      ffi.nullptr, 
      0
    );

    messagesPtr.free(messagesCopy.length);
    calloc.free(promptTokens);
    Llama.lib.llama_batch_free(batch);
  }

  @override
  void stop() {
    _completer?.complete();
  }

  @override
  void free() {
    Llama.lib.llama_free_model(_model);
    Llama.lib.llama_sampler_free(_sampler);
    Llama.lib.llama_free(_context);
  }
}