part of '../llama.dart';

enum IsolateCommand {
  stop,
  clear;
}

typedef ErrorResponse = ({
  String message
});

typedef IsolateArguments = ({
  InitArguments init,
  SendPort sendPort
});

class LlamaCPP {
  static LlamaCppNative? _llamaCppNative;
  StreamController<String> _responseController = StreamController<String>()..close();
  SendPort? _sendPort;

  LlamaCPP(String modelPath, ModelParams modelParams, ContextParams contextParams, SamplingParams samplingParams) {
    final initParams = (
      modelPath: modelPath,
      modelParams: modelParams,
      contextParams: contextParams,
      samplingParams: samplingParams
    );

    _listener(initParams);
  }

  void _listener(InitArguments args) async {
    final receivePort = ReceivePort();
    final completer = Completer();

    final isolateParams = (
      init: args,
      sendPort: receivePort.sendPort
    );

    await Isolate.spawn(_isolate, isolateParams);

    await for (var data in receivePort) {
      if (data is ErrorResponse) {
        completer.completeError(Exception(data.message));
      } 
      else if (data is String) {
        _responseController.add(data);
      }
      else if (data is SendPort) {
        _sendPort = data;
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

  void _isolate(IsolateArguments args) async {
    try {
      _llamaCppNative = LlamaCppNative.fromInitArguments(args.init);
      _sendPort = args.sendPort;
      _isolateListener();
    }
    catch (e) {
      args.sendPort.send((message: e.toString()));
    }
  }

  void _isolateListener() async {
    final receivePort = ReceivePort();
    _sendPort!.send(receivePort.sendPort);

    await for (var data in receivePort) {
      if (data is IsolateCommand) {
        _isolateCommand(data);
      }
      else if (data is List<ChatMessage>) {
        _isolatePrompt(data);
      }
    }
  }

  void _isolateCommand(IsolateCommand command) {
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

  void _isolatePrompt(List<ChatMessage> messages) async {
    final response = _llamaCppNative!.prompt(messages);

    await for (var message in response) {
      _sendPort!.send(message);
    }

    _sendPort!.send(null);
  }

  Stream<String> prompt(List<ChatMessage> messages) async* {   
    _responseController = StreamController<String>();
    
    _sendPort!.send(messages);

    await for (var message in _responseController.stream) {
      yield message;
    }
  }

  void stop() {
    _sendPort!.send(IsolateCommand.stop);
  }

  void clear() {
    _sendPort!.send(IsolateCommand.clear);
  }
}
