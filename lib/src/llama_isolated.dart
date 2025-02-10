part of '../lcpp.dart';

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
      if (data is List<_ChatMessageRecord>) {
        final messages = ChatMessages._fromRecords(data);
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

/// A class that isolates the Llama implementation to run in a separate isolate.
///
/// This class implements the [Llama] interface and provides methods to interact
/// with the Llama model in an isolated environment.
///
/// The [LlamaIsolated] constructor initializes the isolate with the provided
/// model, context, and sampling parameters.
///
/// The [_listener] method listens for messages from the isolate and handles
/// them accordingly. It adds responses to the [_responseController] stream,
/// sets the [_sendPort] when received, and completes the [_initialized] completer.
///
/// The [prompt] method sends a list of [ChatMessage] to the isolate and returns
/// a stream of responses. It waits for the isolate to be initialized before
/// sending the messages.
///
/// The [stop] method sends a signal to the isolate to stop processing. It waits
/// for the isolate to be initialized before sending the signal.
///
/// The [free] method sends a signal to the isolate to free resources. It waits
/// for the isolate to be initialized before sending the signal.
class LlamaIsolated implements Llama {
  final Completer _initialized = Completer();
  StreamController<String> _responseController = StreamController<String>()..close();
  SendPort? _sendPort;

  /// Constructs an instance of [LlamaIsolated].
  ///
  /// Initializes the [LlamaIsolated] with the provided parameters and sets up
  /// the listener.
  ///
  /// Parameters:
  /// - [modelParams]: The parameters required for the model. This parameter is required.
  /// - [contextParams]: The parameters for the context. This parameter is optional and defaults to an instance of [ContextParams].
  /// - [samplingParams]: The parameters for sampling. This parameter is optional and defaults to an instance of [SamplingParams] with `greedy` set to `true`.
  LlamaIsolated({
    required ModelParams modelParams, 
    ContextParams contextParams = const ContextParams(),
    SamplingParams samplingParams = const SamplingParams(greedy: true)
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
    
    _sendPort!.send(messages._toRecords());

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