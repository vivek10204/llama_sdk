part of 'package:lcpp/lcpp.dart';

typedef _LlamaWorkerRecord = (SendPort, String);

class _LlamaWorkerParams {
  final SendPort sendPort;
  final LlamaController llamaController;

  _LlamaWorkerParams({required this.sendPort, required this.llamaController});

  _LlamaWorkerRecord toRecord() {
    return (sendPort, llamaController.toJson());
  }
}
