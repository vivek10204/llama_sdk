part of '../llama.dart';

typedef ChatMessageRecord = (
  String role,
  String content
);

class ChatMessage {
  final String role;
  final String content;

  ChatMessage({
    required this.role,
    required this.content,
  });

  ChatMessage.fromRecord(ChatMessageRecord record)
      : role = record.$1,
        content = record.$2;

  ChatMessage.fromNative(llama_chat_message message)
      : role = message.role.cast<Utf8>().toDartString(),
        content = message.content.cast<Utf8>().toDartString();

  llama_chat_message toNative() {
    final message = calloc<llama_chat_message>();
    message.ref.role = role.toNativeUtf8().cast<ffi.Char>();
    message.ref.content = content.toNativeUtf8().cast<ffi.Char>();

    return message.ref;
  }

  ChatMessageRecord toRecord() => (
    role,
    content
  );
}

extension ChatMessages on List<ChatMessage> {
  static List<ChatMessage> fromRecords(List<ChatMessageRecord> records) {
    final List<ChatMessage> messages = [];

    for (var record in records) {
      messages.add(ChatMessage.fromRecord(record));
    }

    return messages;
  }

  ffi.Pointer<llama_chat_message> toNative() {
    final messages = calloc<llama_chat_message>(length);

    for (var i = 0; i < length; i++) {
      messages[i] = this[i].toNative();
    }

    return messages;
  }

  List<ChatMessageRecord> toRecords() {
    final List<ChatMessageRecord> records = [];

    for (var i = 0; i < length; i++) {
      records.add(this[i].toRecord());
    }

    return records;
  }
}