part of '../llama.dart';

typedef _IsolateArguments = ({
  ModelParams modelParams,
  ContextParams contextParams,
  SamplingParams samplingParams,
  SendPort sendPort
});

extension _IsolateArgumentsExtension on _IsolateArguments {
  _SerializableIsolateArguments get toSerializable => (
    modelParams.toJson(),
    contextParams.toJson(),
    samplingParams.toJson(),
    sendPort
  );
}

typedef _SerializableIsolateArguments = (
  String,
  String,
  String,
  SendPort
);

extension _SerializableIsolateArgumentsExtension on _SerializableIsolateArguments {
  ModelParams get modelParams => ModelParams.fromJson(this.$1);

  ContextParams get contextParams => ContextParams.fromJson(this.$2);

  SamplingParams get samplingParams => SamplingParams.fromJson(this.$3);

  SendPort get sendPort => this.$4;
}

void _isolateEntry(_SerializableIsolateArguments args) async {
  final SendPort sendPort = args.sendPort;
  final LlamaNative llamaCppNative;

  try {
    final receivePort = ReceivePort();
    sendPort.send(receivePort.sendPort);

    llamaCppNative = LlamaNative(
      modelParams: args.modelParams,
      contextParams: args.contextParams,
      samplingParams: args.samplingParams
    );

    await for (final data in receivePort) {
      if (data is List<ChatMessageRecord>) {
        final messages = ChatMessages.fromRecords(data);
        final stream = llamaCppNative.prompt(messages);

        await for (final response in stream) {
          sendPort.send(response);
        }

        sendPort.send(false);
      }
      else if (data is bool) {
        if (data) {
          llamaCppNative.free();
          return;
        }
        else {
          llamaCppNative.stop();
        }

        sendPort.send(data);
      }
    }
  }
  catch (e) {
    log('LlamaIsolateEntry: $e');
  }
}

class LlamaIsolated implements Llama {
  final Completer _initialized = Completer();
  StreamController<String> _responseController = StreamController<String>()..close();
  SendPort? _sendPort;

  LlamaIsolated({
    required ModelParams modelParams, 
    ContextParams contextParams = const ContextParams(),
    SamplingParams samplingParams = const SamplingParams()
  }) {
    _listener(modelParams, contextParams, samplingParams);
  }

  void _listener(ModelParams modelParams, ContextParams contextParams, SamplingParams samplingParams) async {
    final receivePort = ReceivePort();

    final isolateParams = (
      modelParams: modelParams,
      contextParams: contextParams,
      samplingParams: samplingParams,
      sendPort: receivePort.sendPort
    );

    await Isolate.spawn(_isolateEntry, isolateParams.toSerializable);

    await for (final data in receivePort) {
      if (data is String) {
         _responseController.add(data);
      } 
      else if (data is SendPort) {
        _sendPort = data;
        _initialized.complete();
      }
      else if (data is bool) {
        _responseController.close();
        
        if (data) return;
      }
    }
  }

  @override
  Stream<String> prompt(List<ChatMessage> messages) async* {  
    if (!_initialized.isCompleted) {
      await _initialized.future;
    }

    _responseController = StreamController<String>();
    
    _sendPort!.send(messages.toRecords());

    await for (final response in _responseController.stream) {
      yield response;
    }
  }

  @override
  void stop() async {
    if (!_initialized.isCompleted) {
      await _initialized.future;
    }

    _sendPort!.send(false);
  }

  @override
  void free() async {
    if (!_initialized.isCompleted) {
      await _initialized.future;
    }

    _sendPort!.send(true);
  }
}