part of '../llama.dart';

typedef OnProgressCallback = bool Function(double progress);

class ModelParams {
  bool? vocabOnly;
  bool? useMmap;
  bool? useMlock;
  bool? checkTensors;

  ModelParams({
    this.vocabOnly,
    this.useMmap,
    this.useMlock,
    this.checkTensors,
  });

  factory ModelParams.fromInt(int buffer) {
    final modelParams = ModelParams();

    if ((buffer & 1 << 0) != 0) {
      modelParams.vocabOnly = (buffer & 1 << 1) != 0;
    }

    if ((buffer & 1 << 2) != 0) {
      modelParams.useMmap = (buffer & 1 << 3) != 0;
    }

    if ((buffer & 1 << 4) != 0) {
      modelParams.useMlock = (buffer & 1 << 5) != 0;
    }

    if ((buffer & 1 << 6) != 0) {
      modelParams.checkTensors = (buffer & 1 << 7) != 0;
    }

    return modelParams;
  }

  llama_model_params toNative() {
    final llama_model_params modelParams = LlamaCppNative.lib.llama_model_default_params();

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

  int toInt() {
    int buffer = 0;

    if (vocabOnly != null) {
      buffer |= 1 << 0;
      buffer |= (vocabOnly! ? 1 : 0) << 1;
    }

    if (useMmap != null) {
      buffer |= 1 << 2;
      buffer |= (useMmap! ? 1 : 0) << 3;
    }

    if (useMlock != null) {
      buffer |= 1 << 4;
      buffer |= (useMlock! ? 1 : 0) << 5;
    }

    if (checkTensors != null) {
      buffer |= 1 << 6;
      buffer |= (checkTensors! ? 1 : 0) << 7;
    }

    return buffer;
  }
}