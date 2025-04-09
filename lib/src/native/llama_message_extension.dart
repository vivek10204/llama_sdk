part of 'package:llama_sdk/llama_sdk.dart';

typedef _LlamaMessageRecord = (String role, String content);

extension _LlamaMessageExtension on LlamaMessage {
  static LlamaMessage fromRecord(_LlamaMessageRecord record) =>
      LlamaMessage.withRole(role: record.$1, content: record.$2);

  _LlamaMessageRecord toRecord() => (role, content);
}

extension _LlamaMessagesExtension on List<LlamaMessage> {
  static List<LlamaMessage> fromRecords(List<_LlamaMessageRecord> records) {
    final List<LlamaMessage> messages = [];

    for (var record in records) {
      messages.add(_LlamaMessageExtension.fromRecord(record));
    }

    return messages;
  }

  List<Map<String, String>> toMapList() =>
      map((message) => message.toMap()).toList();

  String toJson() => jsonEncode(toMapList());

  ffi.Pointer<ffi.Char> toPointer() => toJson().toNativeUtf8().cast<ffi.Char>();

  List<_LlamaMessageRecord> toRecords() {
    final List<_LlamaMessageRecord> records = [];

    for (var i = 0; i < length; i++) {
      records.add(this[i].toRecord());
    }

    return records;
  }
}
