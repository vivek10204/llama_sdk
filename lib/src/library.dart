part of '../llama.dart';

llama? _lib;

llama get lib {
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