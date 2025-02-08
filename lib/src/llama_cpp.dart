part of '../llama.dart';

typedef StringResponse = (
  bool error,
  String message
);

typedef IsolateArguments = (
  String modelPath,
  String modelParams,
  String contextParams,
  String samplingParams,
  SendPort sendPort
);

class LlamaCPP {
  final Completer _initialized = Completer();
  StreamController<String> _responseController = StreamController<String>()..close();
  SendPort? _sendPort;

  LlamaCPP(String modelPath, ModelParams modelParams, ContextParams contextParams, SamplingParams samplingParams) {
    _listener(modelPath, modelParams, contextParams, samplingParams);
  }

  void _listener(String modelPath, ModelParams modelParams, ContextParams contextParams, SamplingParams samplingParams) async {
    final receivePort = ReceivePort();

    final isolateParams = (
      modelPath,
      modelParams.toJson(),
      contextParams.toJson(),
      samplingParams.toJson(),
      receivePort.sendPort
    );

    await Isolate.spawn(entryPoint, isolateParams);

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
        print('Isolate stopped');
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

void entryPoint(IsolateArguments args) async {
  final SendPort sendPort = args.$5;
  final LlamaNative llamaCppNative;

  try {
    final receivePort = ReceivePort();
    sendPort.send(receivePort.sendPort);

    llamaCppNative = LlamaNative(
      modelParams: ModelParams.fromJson(args.$2),
      contextParams: ContextParams.fromJson(args.$3),
      samplingParams: SamplingParams.fromJson(args.$4)
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