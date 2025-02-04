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

    await Isolate.spawn(LlamaCppIsolate.entryPoint, isolateParams);

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

    await for (var message in _responseController.stream) {
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

class LlamaCppIsolate {
  static LlamaCppNative? _llamaCppNative;
  static SendPort? _sendPort;

  static void entryPoint(IsolateArguments args) async {
    _sendPort = args.$5;

    try {
      _llamaCppNative = LlamaCppNative.fromParams(
        args.$1,
        ModelParams.fromJson(args.$2),
        ContextParams.fromJson(args.$3),
        SamplingParams.fromJson(args.$4)
      );

      _isolateListener();
    }
    catch (e) {
      _sendPort!.send((message: e.toString()));
    }
  }

  static void _isolateListener() async {
    final receivePort = ReceivePort();
    _sendPort!.send(receivePort.sendPort);

    await for (var data in receivePort) {
      if (data is IsolateCommand) {
        _isolateCommand(data);
      }
      else if (data is List<ChatMessageRecord>) {
        _isolatePrompt(data);
      }
    }
  }

  static void _isolateCommand(IsolateCommand command) {
    switch (command) {
      case IsolateCommand.stop:
        _llamaCppNative!.stop();
        break;
      case IsolateCommand.clear:
        _llamaCppNative!.clear();
        break;
    }

    _sendPort!.send(command);
  }

  static void _isolatePrompt(List<ChatMessageRecord> messages) async {
    final response = _llamaCppNative!.prompt(ChatMessages.fromRecords(messages));

    await for (var message in response) {
      _sendPort!.send((false, message));
    }

    _sendPort!.send(null);
  }
}