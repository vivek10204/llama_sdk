part of '../llama.dart';

abstract interface class Llama {
  static llama? _lib;

  static llama get lib {
    if (_lib == null) {
      if (Platform.isWindows) {
        _lib = llama(ffi.DynamicLibrary.open('llama.dll'));
      } 
      else if (Platform.isLinux || Platform.isAndroid) {
        _lib = llama(ffi.DynamicLibrary.open('libllama.so'));
      } 
      else if (Platform.isMacOS || Platform.isIOS) {
        _lib = llama(ffi.DynamicLibrary.open('llama.framework/llama'));
      } 
      else {
        throw Exception('Unsupported platform');
      }
    }
    return _lib!;
  }

  factory Llama({
    required ModelParams modelParams, 
    ContextParams contextParams = const ContextParams(),
    SamplingParams samplingParams = const SamplingParams(),
    bool isolate = false,
  }) => isolate ? LlamaIsolated(
    modelParams: modelParams,
    contextParams: contextParams,
    samplingParams: samplingParams,
  ) : LlamaNative(
    modelParams: modelParams,
    contextParams: contextParams,
    samplingParams: samplingParams,
  );

  Stream<String> prompt(List<ChatMessage> messages);

  void stop();

  void free();
}