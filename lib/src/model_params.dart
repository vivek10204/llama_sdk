part of '../llama.dart';

class ModelParams {
  final String path;
  final bool? vocabOnly;
  final bool? useMmap;
  final bool? useMlock;
  final bool? checkTensors;

  const ModelParams({
    required this.path,
    this.vocabOnly,
    this.useMmap,
    this.useMlock,
    this.checkTensors,
  });

  factory ModelParams.fromMap(Map<String, dynamic> map) => ModelParams(
    path: map['path'],
    vocabOnly: map['vocabOnly'],
    useMmap: map['useMmap'],
    useMlock: map['useMlock'],
    checkTensors: map['checkTensors']
  );

  factory ModelParams.fromJson(String source) => ModelParams.fromMap(jsonDecode(source));

  llama_model_params toNative() {
    final llama_model_params modelParams = lib.llama_model_default_params();
    log("Model params initialized");

    if (vocabOnly != null) {
      modelParams.vocab_only = vocabOnly!;
    }

    if (useMmap != null) {
      modelParams.use_mmap = useMmap!;
    }

    if (useMlock != null) {
      modelParams.use_mlock = useMlock!;
    }

    if (checkTensors != null) {
      modelParams.check_tensors = checkTensors!;
    }

    return modelParams;
  }

  Map<String, dynamic> toMap() => {
    'path': path,
    'vocabOnly': vocabOnly,
    'useMmap': useMmap,
    'useMlock': useMlock,
    'checkTensors': checkTensors,
  };

  String toJson() => jsonEncode(toMap());
}