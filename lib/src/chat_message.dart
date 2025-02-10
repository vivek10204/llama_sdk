part of '../lcpp.dart';

typedef _ChatMessageRecord = (
  String role,
  String content
);

abstract class ChatMessage {
  String get role;

  String content;

  ChatMessage(this.content);

  factory ChatMessage.withRole({
    required String role,
    required String content,
  }) {
    switch (role) {
      case 'user':
        return UserChatMessage(content);
      case 'assistant':
        return AssistantChatMessage(content);
      case 'system':
        return SystemChatMessage(content);
      default:
        throw ArgumentError('Invalid role: $role');
    }
  }

  factory ChatMessage._fromRecord(_ChatMessageRecord record) => ChatMessage.withRole(
    role: record.$1, 
    content: record.$2
  );

  factory ChatMessage.fromNative(llama_chat_message message) => ChatMessage.withRole(
    role: message.role.cast<Utf8>().toDartString(),
    content: message.content.cast<Utf8>().toDartString()
  );

  llama_chat_message toNative() {
    final message = calloc<llama_chat_message>();
    message.ref.role = role.toNativeUtf8().cast<ffi.Char>();
    message.ref.content = content.toNativeUtf8().cast<ffi.Char>();

    return message.ref;
  }

  _ChatMessageRecord _toRecord() => (
    role,
    content
  );
}

class UserChatMessage extends ChatMessage {
  @override
  String get role => 'user';

  UserChatMessage(super.content);
}

class AssistantChatMessage extends ChatMessage {
  @override
  String get role => 'assistant';

  AssistantChatMessage(super.content);
}

class SystemChatMessage extends ChatMessage {
  @override
  String get role => 'system';

  SystemChatMessage(super.content);
}

extension ChatMessages on List<ChatMessage> {
  static List<ChatMessage> _fromRecords(List<_ChatMessageRecord> records) {
    final List<ChatMessage> messages = [];

    for (var record in records) {
      messages.add(ChatMessage._fromRecord(record));
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

  List<_ChatMessageRecord> _toRecords() {
    final List<_ChatMessageRecord> records = [];

    for (var i = 0; i < length; i++) {
      records.add(this[i]._toRecord());
    }

    return records;
  }

  List<ChatMessage> copy() {
    final List<ChatMessage> messages = [];

    for (var message in this) {
      messages.add(ChatMessage.withRole(role: message.role, content: message.content));
    }

    return messages;
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