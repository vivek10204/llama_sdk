/// An abstract class representing a chat message.
///
/// This class provides a base for different types of chat messages, each with
/// a specific role and content.
///
/// Properties:
/// - `role`: A string representing the role of the message sender.
/// - `content`: The content of the message.
///
/// Constructors:
/// - `LlamaMessage(String content)`: Initializes a chat message with the given content.
/// - `LlamaMessage.withRole({required String role, required String content})`: Factory constructor
///   that creates a chat message based on the specified role. Throws an `ArgumentError` if the role is invalid.
/// - `LlamaMessage._fromRecord(_LlamaMessageRecord record)`: Factory constructor that creates a chat message
///   from a record.
/// - `LlamaMessage.fromNative(llama_chat_message message)`: Factory constructor that creates a chat message
///   from a native `llama_chat_message`.
///
/// Methods:
/// - `llama_chat_message toNative()`: Converts the chat message to a native `llama_chat_message`.
/// - `_LlamaMessageRecord _toRecord()`: Converts the chat message to a record.
abstract class LlamaMessage {
  /// The role of the chat message sender.
  ///
  /// This property represents the role of the entity that sent the chat message,
  /// such as 'user', 'assistant', or any other defined role.
  String get role;

  /// The content of the chat message.
  String content;

  /// Creates a new instance of [LlamaMessage] with the given content.
  ///
  /// The [content] parameter represents the message content.
  LlamaMessage(this.content);

  /// Factory constructor to create a `LlamaMessage` instance based on the provided role.
  ///
  /// The `role` parameter determines the type of `LlamaMessage` to create:
  /// - 'user': Creates a `UserLlamaMessage`.
  /// - 'assistant': Creates an `AssistantLlamaMessage`.
  /// - 'system': Creates a `SystemLlamaMessage`.
  ///
  /// Throws an [ArgumentError] if the provided role is not one of the expected values.
  ///
  /// Parameters:
  /// - `role`: The role of the chat message (e.g., 'user', 'assistant', 'system').
  /// - `content`: The content of the chat message.
  factory LlamaMessage.withRole({
    required String role,
    required String content,
  }) {
    switch (role) {
      case 'user':
        return UserLlamaMessage(content);
      case 'assistant':
        return AssistantLlamaMessage(content);
      case 'system':
        return SystemLlamaMessage(content);
      default:
        throw ArgumentError('Invalid role: $role');
    }
  }

  /// Creates a [LlamaMessage] instance from a map.
  ///
  /// The map should contain a 'role' key with one of the following values:
  /// - 'user': to create a [UserLlamaMessage]
  /// - 'assistant': to create an [AssistantLlamaMessage]
  /// - 'system': to create a [SystemLlamaMessage]
  ///
  /// The map should also contain a 'content' key with the message content.
  ///
  /// Throws an [Exception] if the 'role' value is invalid.
  factory LlamaMessage.fromMap(Map<String, dynamic> map) {
    switch (map['role']) {
      case 'user':
        return UserLlamaMessage(map['content']);
      case 'assistant':
        return AssistantLlamaMessage(map['content']);
      case 'system':
        return SystemLlamaMessage(map['content']);
      default:
        throw Exception('Invalid role');
    }
  }

  /// Converts a [LlamaMessage] object to a map representation.
  ///
  /// The returned map contains the following key-value pairs:
  /// - 'role': The role of the message sender.
  /// - 'content': The content of the message.
  ///
  /// Example:
  /// ```dart
  /// LlamaMessage message = LlamaMessage(role: 'user', content: 'Hello');
  /// Map<String, String> map = messageToMap(message);
  /// // map = {'role': 'user', 'content': 'Hello'}
  /// ```
  ///
  /// [message]: The [LlamaMessage] object to be converted.
  static Map<String, String> messageToMap(LlamaMessage message) => {
        'role': message.role,
        'content': message.content,
      };

  /// Converts the current `LlamaMessage` instance to a map.
  ///
  /// This method uses the `messageToMap` function to transform the
  /// `LlamaMessage` object into a `Map<String, String>`, which can be
  /// useful for serialization or other operations that require a map
  /// representation of the message.
  ///
  /// Returns a `Map<String, String>` representation of the `LlamaMessage`.
  Map<String, String> toMap() => messageToMap(this);
}

/// A class representing a chat message from a user.
///
/// This class extends the [LlamaMessage] class and overrides the [role] getter
/// to return 'user', indicating that the message is from a user.
///
/// Example usage:
/// ```dart
/// var message = UserLlamaMessage('Hello, world!');
/// print(message.role); // Output: user
/// print(message.content); // Output: Hello, world!
/// ```
///
/// The [UserLlamaMessage] constructor takes the content of the message as a parameter.
class UserLlamaMessage extends LlamaMessage {
  @override
  String get role => 'user';

  /// A class representing a user chat message.
  ///
  /// This class extends a base class and takes the content of the message as a parameter.
  ///
  /// Example usage:
  /// ```dart
  /// var message = UserLlamaMessage('Hello, world!');
  /// print(message.content); // Outputs: Hello, world!
  /// ```
  ///
  /// Parameters:
  /// - `content`: The content of the chat message.
  UserLlamaMessage(super.content);
}

/// A class representing a chat message from an assistant.
///
/// This class extends the [LlamaMessage] class and overrides the [role] getter
/// to return 'assistant', indicating that the message is from an assistant.
///
/// Example usage:
/// ```dart
/// var message = AssistantLlamaMessage('Hello, how can I assist you?');
/// print(message.role); // Outputs: assistant
/// print(message.content); // Outputs: Hello, how can I assist you?
/// ```
///
/// See also:
/// - [LlamaMessage], the base class for chat messages.
class AssistantLlamaMessage extends LlamaMessage {
  @override
  String get role => 'assistant';

  /// Represents a chat message from the assistant.
  ///
  /// The [AssistantLlamaMessage] class extends a base class with the provided
  /// content of the message.
  ///
  /// Example usage:
  /// ```dart
  /// var message = AssistantLlamaMessage('Hello, how can I assist you?');
  /// print(message.content); // Output: Hello, how can I assist you?
  /// ```
  ///
  /// Parameters:
  /// - `content`: The content of the chat message.
  AssistantLlamaMessage(super.content);
}

/// A class representing a system-generated chat message.
///
/// This class extends the [LlamaMessage] class and overrides the `role`
/// property to return 'system', indicating that the message is generated
/// by the system.
///
/// Example usage:
/// ```dart
/// var systemMessage = SystemLlamaMessage('System maintenance scheduled.');
/// print(systemMessage.role); // Outputs: system
/// print(systemMessage.content); // Outputs: System maintenance scheduled.
/// ```
///
/// The [SystemLlamaMessage] constructor takes a single parameter, `content`,
/// which represents the content of the chat message.
class SystemLlamaMessage extends LlamaMessage {
  @override
  String get role => 'system';

  /// Represents a system-generated chat message.
  ///
  /// This class extends the base class and takes the content of the message as a parameter.
  ///
  /// Example usage:
  /// ```dart
  /// var message = SystemLlamaMessage('System maintenance scheduled.');
  /// print(message.content); // Outputs: System maintenance scheduled.
  /// ```
  ///
  /// Parameters:
  /// - `content`: The content of the chat message.
  SystemLlamaMessage(super.content);
}

/// An extension on the `List<LlamaMessage>` class to provide additional functionality.
extension LlamaMessages on List<LlamaMessage> {
  /// Creates a copy of the list of `LlamaMessage` objects.
  ///
  /// This method iterates over the current list of `LlamaMessage` instances,
  /// creates a new `LlamaMessage` for each one with the same role and content,
  /// and returns a new list containing these copied messages.
  ///
  /// Returns:
  ///   A new list of `LlamaMessage` objects with the same role and content as the original list.
  List<LlamaMessage> copy() {
    final List<LlamaMessage> messages = [];

    for (var message in this) {
      messages.add(
        LlamaMessage.withRole(role: message.role, content: message.content),
      );
    }

    return messages;
  }
}
