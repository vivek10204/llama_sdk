part of 'package:lcpp/lcpp.dart';

typedef _ChatMessageRecord = (String role, String content);

extension _ChatMessageExtension on ChatMessage {
  static ChatMessage fromRecord(_ChatMessageRecord record) =>
      ChatMessage.withRole(role: record.$1, content: record.$2);

  llama_chat_message toNative() {
    final message = calloc<llama_chat_message>();
    message.ref.role = role.toNativeUtf8().cast<ffi.Char>();
    message.ref.content = content.toNativeUtf8().cast<ffi.Char>();

    return message.ref;
  }

  _ChatMessageRecord toRecord() => (role, content);
}

extension _ChatMessagesExtension on List<ChatMessage> {
  static List<ChatMessage> fromRecords(List<_ChatMessageRecord> records) {
    final List<ChatMessage> messages = [];

    for (var record in records) {
      messages.add(_ChatMessageExtension.fromRecord(record));
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

  List<_ChatMessageRecord> toRecords() {
    final List<_ChatMessageRecord> records = [];

    for (var i = 0; i < length; i++) {
      records.add(this[i].toRecord());
    }

    return records;
  }
}

extension _LlamaChatMessagePtrExtension on ffi.Pointer<llama_chat_message> {
  void free(int length) {
    for (var i = 0; i < length; i++) {
      calloc.free(this[i].role);
      calloc.free(this[i].content);
    }

    calloc.free(this);
  }
}
