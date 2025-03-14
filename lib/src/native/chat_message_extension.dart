part of 'package:lcpp/lcpp.dart';

typedef _ChatMessageRecord = (String role, String content);

extension _ChatMessageExtension on ChatMessage {
  static ChatMessage fromRecord(_ChatMessageRecord record) =>
      ChatMessage.withRole(role: record.$1, content: record.$2);

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

  List<Map<String, String>> toMapList() => map((message) => message.toMap()).toList();

  String toJson() => jsonEncode(toMapList());

  ffi.Pointer<ffi.Char> toPointer() => toJson().toNativeUtf8().cast<ffi.Char>();

  List<_ChatMessageRecord> toRecords() {
    final List<_ChatMessageRecord> records = [];

    for (var i = 0; i < length; i++) {
      records.add(this[i].toRecord());
    }

    return records;
  }
}
