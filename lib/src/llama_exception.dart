part of '../llama.dart';

class LlamaException implements Exception {
  final String message;

  LlamaException(this.message);

  @override
  String toString() => message;
}