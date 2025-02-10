part of '../llama.dart';

typedef IsolateArguments = ({
  ModelParams modelParams,
  ContextParams contextParams,
  SamplingParams samplingParams,
  SendPort sendPort
});

extension IsolateArgumentsExtension on IsolateArguments {
  SerializableIsolateArguments get toSerializable => (
    modelParams.toJson(),
    contextParams.toJson(),
    samplingParams.toJson(),
    sendPort
  );
}

typedef SerializableIsolateArguments = (
  String,
  String,
  String,
  SendPort
);

extension SerializableIsolateArgumentsExtension on SerializableIsolateArguments {
  ModelParams get modelParams => ModelParams.fromJson(this.$1);

  ContextParams get contextParams => ContextParams.fromJson(this.$2);

  SamplingParams get samplingParams => SamplingParams.fromJson(this.$3);

  SendPort get sendPort => this.$4;
}

class LlamaIsolateEntry {
  static void entry(SerializableIsolateArguments args) async {
    final SendPort sendPort = args.sendPort;
    final LlamaNative llamaCppNative;

    lib.llama_log_set(ffi.Pointer.fromFunction(_logCallback), ffi.nullptr);

    try {
      final receivePort = ReceivePort();
      sendPort.send(receivePort.sendPort);

      llamaCppNative = LlamaNative(
        modelParams: args.modelParams,
        contextParams: args.contextParams,
        samplingParams: args.samplingParams
      );

      await for (var data in receivePort) {
        if (data is List<ChatMessageRecord>) {

          final messages = ChatMessages.fromRecords(data);

          final response = llamaCppNative.prompt(messages);

          await for (var message in response) {
            sendPort.send((false, message));
          }

          sendPort.send(null);
        }
        else if (data is bool) {
          sendPort.send(data);
        }
      }
    }
    catch (e) {
      sendPort.send((message: e.toString()));
    }
  }

  static void _logCallback(int level, ffi.Pointer<ffi.Char> text, ffi.Pointer<ffi.Void> userData) {
    log(text.cast<Utf8>().toDartString());
  }
}