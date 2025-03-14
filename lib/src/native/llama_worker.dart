part of 'package:lcpp/lcpp.dart';

class _LlamaWorker {
  static final _modelFinalizer =
      Finalizer<ffi.Pointer<llama_model>>(lib.llama_free_model);
  static final _contextFinalizer =
      Finalizer<ffi.Pointer<llama_context>>(lib.llama_free);
  static final _samplerFinalizer =
      Finalizer<ffi.Pointer<llama_sampler>>(lib.llama_sampler_free);

  ffi.Pointer<llama_model> _model = ffi.nullptr;
  ffi.Pointer<llama_context> _context = ffi.nullptr;
  ffi.Pointer<llama_sampler> _sampler = ffi.nullptr;

  LlamaController _llamaController;

  int _contextLength = 0;

  final Completer<void> completer = Completer<void>();
  final ReceivePort receivePort = ReceivePort();
  final SendPort sendPort;

  _LlamaWorker({
    required this.sendPort, 
    required LlamaController llamaController
  }) : _llamaController = llamaController {
    lib.ggml_backend_load_all();
    lib.llama_backend_init();

    _init();

    sendPort.send(receivePort.sendPort);
    receivePort.listen(handleData);
  }

  factory _LlamaWorker.fromRecord(_LlamaWorkerRecord record) => _LlamaWorker(
      sendPort: record.$1, llamaController: LlamaController.fromJson(record.$2));

  void _init() {
    final nativeModelParams = _llamaController.getModelParams();
    final nativeModelPath =
        _llamaController.modelFile.path.toNativeUtf8().cast<ffi.Char>();

    if (_model != ffi.nullptr) {
      lib.llama_free_model(_model);
    }

    _model = lib
        .llama_load_model_from_file(nativeModelPath, nativeModelParams);
    assert(_model != ffi.nullptr, LlamaException('Failed to load model'));

    _modelFinalizer.attach(this, _model);

    assert(_model != ffi.nullptr, LlamaException('Model is not loaded'));

    final nativeContextParams = _llamaController.getContextParams();

    if (_context != ffi.nullptr) {
      lib.llama_free(_context);
    }

    _context = lib.llama_init_from_model(_model, nativeContextParams);
    assert(_context != ffi.nullptr,
        LlamaException('Failed to initialize context'));

    _contextFinalizer.attach(this, _context);

    if (_sampler != ffi.nullptr) {
      lib.llama_sampler_free(_sampler);
    }

    final vocab = lib.llama_model_get_vocab(_model);
    _sampler = _llamaController.getSampler(vocab);
    assert(_sampler != ffi.nullptr,
        LlamaException('Failed to initialize sampler'));

    _samplerFinalizer.attach(this, _sampler);
  }

  void handleData(dynamic data) async {
    switch (data.runtimeType) {
      case const (List<_ChatMessageRecord>):
        handlePrompt(data.cast<_ChatMessageRecord>());
        break;
      default:
        completer.completeError(LlamaException('Invalid data type'));
        break;
    }
  }

  void handlePrompt(List<_ChatMessageRecord> data) async {
    final messages = _ChatMessagesExtension.fromRecords(data);
    final stream = prompt(messages);

    await for (final response in stream) {
      sendPort.send(response);
    }

    await Future.delayed(const Duration(milliseconds: 100));

    sendPort.send(null);
  }

  Stream<String> prompt(List<ChatMessage> messages) async* {
    final messagesCopy = messages.copy();

    assert(_model != ffi.nullptr, LlamaException('Model is not loaded'));
    assert(
        _context != ffi.nullptr, LlamaException('Context is not initialized'));
    assert(
        _sampler != ffi.nullptr, LlamaException('Sampler is not initialized'));

    final nCtx = lib.llama_n_ctx(_context);

    ffi.Pointer<ffi.Char> formatted = calloc<ffi.Char>(nCtx);

    final template = lib.llama_model_chat_template(_model, ffi.nullptr);

    ffi.Pointer<llama_chat_message> messagesPtr = messagesCopy.toNative();

    int newContextLength = lib.llama_chat_apply_template(
        template, messagesPtr, messagesCopy.length, true, formatted, nCtx);

    if (newContextLength > nCtx) {
      calloc.free(formatted);
      formatted = calloc<ffi.Char>(newContextLength);
      newContextLength = lib.llama_chat_apply_template(template,
          messagesPtr, messagesCopy.length, true, formatted, newContextLength);
    }

    messagesPtr.free(messagesCopy.length);

    if (newContextLength < 0) {
      throw Exception('Failed to apply template');
    }

    final prompt =
        formatted.cast<Utf8>().toDartString().substring(_contextLength);
    calloc.free(formatted);

    final vocab = lib.llama_model_get_vocab(_model);
    final isFirst = lib.llama_get_kv_cache_used_cells(_context) == 0;

    final promptPtr = prompt.toNativeUtf8().cast<ffi.Char>();

    final nPromptTokens = -lib.llama_tokenize(
        vocab, promptPtr, prompt.length, ffi.nullptr, 0, isFirst, true);
    ffi.Pointer<llama_token> promptTokens = calloc<llama_token>(nPromptTokens);

    if (lib.llama_tokenize(vocab, promptPtr, prompt.length, promptTokens,
            nPromptTokens, isFirst, true) <
        0) {
      throw Exception('Failed to tokenize');
    }

    calloc.free(promptPtr);

    llama_batch batch =
        lib.llama_batch_get_one(promptTokens, nPromptTokens);
    ffi.Pointer<llama_token> newTokenId = calloc<llama_token>(1);

    String response = '';

    while (true) {
      final nCtx = lib.llama_n_ctx(_context);
      final nCtxUsed = lib.llama_get_kv_cache_used_cells(_context);

      if (nCtxUsed + batch.n_tokens > nCtx) {
        throw Exception('Context size exceeded');
      }

      if (lib.llama_decode(_context, batch) != 0) {
        throw Exception('Failed to decode');
      }

      newTokenId.value = lib.llama_sampler_sample(_sampler, _context, -1);

      // is it an end of generation?
      if (lib.llama_vocab_is_eog(vocab, newTokenId.value)) {
        break;
      }

      final buffer = calloc<ffi.Char>(256);
      final n = lib
          .llama_token_to_piece(vocab, newTokenId.value, buffer, 256, 0, true);
      if (n < 0) {
        throw Exception('Failed to convert token to piece');
      }

      final piece = buffer.cast<Utf8>().toDartString();
      calloc.free(buffer);
      response += piece;
      yield piece;

      batch = lib.llama_batch_get_one(newTokenId, 1);
    }

    messagesCopy.add(AssistantChatMessage(response));

    messagesPtr = messagesCopy.toNative();

    _contextLength = lib.llama_chat_apply_template(
        template, messagesPtr, messagesCopy.length, false, ffi.nullptr, 0);

    messagesPtr.free(messagesCopy.length);
    calloc.free(promptTokens);
    lib.llama_batch_free(batch);
  }

  static void entry(_LlamaWorkerRecord record) async {
    final worker = _LlamaWorker.fromRecord(record);
    await worker.completer.future;
  }
}
