part of '../llama.dart';

enum IsolateCommand {
  stop,
  clear;
}

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
      else if (data is IsolateCommand) {
        switch (data) {
          case IsolateCommand.stop:
            break;
          case IsolateCommand.clear:
            break;
        }
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

    _sendPort!.send(IsolateCommand.stop);
  }

  void clear() async {
    if (!_initialized.isCompleted) {
      await _initialized.future;
    }

    _sendPort!.send(IsolateCommand.clear);
  }
}

void entryPoint(IsolateArguments args) async {
  print("Isolate started");
  final SendPort sendPort = args.$5;
  final LlamaCppNative llamaCppNative;

  try {
    final receivePort = ReceivePort();
    sendPort.send(receivePort.sendPort);

    llamaCppNative = LlamaCppNative.fromParams(
      args.$1,
      ModelParams.fromJson(args.$2),
      ContextParams.fromJson(args.$3),
      SamplingParams.fromJson(args.$4)
    );

    print("LlamaCppNative created");

    await for (var data in receivePort) {
      if (data is List<ChatMessageRecord>) {
        print("Received messages");

        final messages = ChatMessages.fromRecords(data);

        final response = llamaCppNative.prompt(messages);

        await for (var message in response) {
          sendPort.send((false, message));
        }

        sendPort.send(null);
      }
      else if (data is IsolateCommand) {
        switch (data) {
          case IsolateCommand.stop:
            llamaCppNative.stop();
            break;
          case IsolateCommand.clear:
            llamaCppNative.clear();
            break;
        }

        sendPort.send(data);
      }
    }
  }
  catch (e) {
    sendPort.send((message: e.toString()));
  }
}