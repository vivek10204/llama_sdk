part of '../llama.dart';

typedef StringResponse = (
  bool error,
  String message
);

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

    await Isolate.spawn(LlamaIsolateEntry.entry, isolateParams.toSerializable);

    await for (var data in receivePort) {
      if (data is String) {
         _responseController.add(data);
      } 
      else if (data is SendPort) {
        _sendPort = data;
        _initialized.complete();
      }
      else if (data == null) {
        _responseController.close();
        break;
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

    _responseController.close();
  }

  @override
  void stop() async {
    if (!_initialized.isCompleted) {
      await _initialized.future;
    }

    _sendPort!.send(null);
  }
}