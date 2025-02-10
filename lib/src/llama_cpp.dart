part of '../llama.dart';

typedef StringResponse = (
  bool error,
  String message
);

class LlamaCPP {
  final Completer _initialized = Completer();
  StreamController<String> _responseController = StreamController<String>()..close();
  SendPort? _sendPort;

  LlamaCPP({required ModelParams modelParams, required ContextParams contextParams, required SamplingParams samplingParams}) {
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
      if (data is StringResponse) {
        if (data.$1) {
          throw Exception(data.$2);
        }
        else {
          _responseController.add(data.$2);
        }
      } 
      else if (data is SendPort) {
        _sendPort = data;
        _initialized.complete();
      }
      else if (data is bool) {
        log('Isolate stopped');
      }
      else if (data == null) {
        _responseController.close();
      }
    }
  }

  Stream<String> prompt(List<ChatMessage> messages) async* {  
    if (!_initialized.isCompleted) {
      await _initialized.future;
    }

    _responseController = StreamController<String>();
    
    _sendPort!.send(messages.toRecords());

    await for (final message in _responseController.stream) {
      yield message;
    }
  }

  void stop() async {
    if (!_initialized.isCompleted) {
      await _initialized.future;
    }

    _sendPort!.send(true);
  }
}