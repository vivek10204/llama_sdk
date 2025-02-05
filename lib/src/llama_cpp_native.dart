part of '../llama.dart';

class LlamaCppNative {
  final ffi.Pointer<llama_model> _model;
  final ffi.Pointer<llama_context> _context;
  final ffi.Pointer<llama_sampler> _sampler;

  Completer? _completer;
  int _contextLength = 0;

  LlamaCppNative({
    required ffi.Pointer<llama_model> model, 
    required ffi.Pointer<llama_context> context, 
    required ffi.Pointer<llama_sampler> sampler
  }) : _model = model, _context = context, _sampler = sampler;

  factory LlamaCppNative.fromParams(
    String modelPath, 
    ModelParams modelParams, 
    ContextParams contextParams, 
    SamplingParams samplingParams
  ) {
    lib.ggml_backend_load_all();

    final nativeModelParams = modelParams.toNative();
      
    final model = lib.llama_load_model_from_file(
      modelPath.toNativeUtf8().cast<ffi.Char>(), 
      nativeModelParams
    );
    assert(model != ffi.nullptr, 'Failed to load model');

    final nativeContextParams = contextParams.toNative();

    final context = lib.llama_init_from_model(model, nativeContextParams);
    assert(context != ffi.nullptr, 'Failed to initialize context');

    final vocab = lib.llama_model_get_vocab(model);
    final sampler = samplingParams.toNative(vocab);
    assert(sampler != ffi.nullptr, 'Failed to initialize sampler');

    return LlamaCppNative(model: model, context: context, sampler: sampler);
  }

  Stream<String> prompt(List<ChatMessage> messages) async* {
    assert(_model != ffi.nullptr, 'Model is not loaded');
    assert(_context != ffi.nullptr, 'Context is not initialized');
    assert(_sampler != ffi.nullptr, 'Sampler is not initialized');

    _completer = Completer();

    final nCtx = lib.llama_n_ctx(_context);

    ffi.Pointer<ffi.Char> formatted = calloc<ffi.Char>(nCtx);

    final template = lib.llama_model_chat_template(_model, ffi.nullptr);

    int newContextLength = lib.llama_chat_apply_template(
      template, 
      messages.toNative(), 
      messages.length, 
      true, 
      formatted, 
      nCtx
    );

    if (newContextLength > nCtx) {
      formatted = calloc<ffi.Char>(newContextLength);
      newContextLength = lib.llama_chat_apply_template(
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

    final prompt = formatted.cast<Utf8>().toDartString().substring(_contextLength);

    final generation = _generate(prompt);

    String finalOutput = '';

    await for (final piece in generation) {
      finalOutput += piece;
      yield piece;
    }

    messages.add(ChatMessage(
      role: 'assistant',
      content: finalOutput
    ));

    _contextLength = lib.llama_chat_apply_template(
      template, 
      messages.toNative(), 
      messages.length, 
      false, 
      ffi.nullptr, 
      0
    );
  }

  Stream<String> _generate(String prompt) async* {
    final vocab = lib.llama_model_get_vocab(_model);
    final isFirst = lib.llama_get_kv_cache_used_cells(_context) == 0;

    final nPromptTokens = -lib.llama_tokenize(vocab, prompt.toNativeUtf8().cast<ffi.Char>(), prompt.length, ffi.nullptr, 0, isFirst, true);
    ffi.Pointer<llama_token> promptTokens = calloc<llama_token>(nPromptTokens);

    if (lib.llama_tokenize(vocab, prompt.toNativeUtf8().cast<ffi.Char>(), prompt.length, promptTokens, nPromptTokens, isFirst, true) < 0) {
      throw Exception('Failed to tokenize');
    }

    llama_batch batch = lib.llama_batch_get_one(promptTokens, nPromptTokens);
    int newTokenId;
    
    while (!_completer!.isCompleted) {
      final nCtx = lib.llama_n_ctx(_context);
      final nCtxUsed = lib.llama_get_kv_cache_used_cells(_context);

      if (nCtxUsed + batch.n_tokens > nCtx) {
        throw Exception('Context size exceeded');
      }

      if (lib.llama_decode(_context, batch) != 0) {
        throw Exception('Failed to decode');
      }

      newTokenId = lib.llama_sampler_sample(_sampler, _context, -1);

      // is it an end of generation?
      if (lib.llama_vocab_is_eog(vocab, newTokenId)) {
        break;
      }

      final buffer = calloc<ffi.Char>(256);
      final n = lib.llama_token_to_piece(vocab, newTokenId, buffer, 256, 0, true);
      if (n < 0) {
        throw Exception('Failed to convert token to piece');
      }

      yield buffer.cast<Utf8>().toDartString();

      final newTokenPointer = calloc<llama_token>(1);
      newTokenPointer.value = newTokenId;

      batch = lib.llama_batch_get_one(newTokenPointer, 1);
    }
  }

  void stop() {
    _completer?.complete();
  }

  void clear() {
    _contextLength = 0;
  }

  void free() {
    lib.llama_sampler_free(_sampler);
    lib.llama_free(_context);
    lib.llama_free_model(_model);
  }
}