import 'dart:math';

import 'package:flutter/material.dart';
import 'dart:async';

import 'package:file_picker/file_picker.dart';
import 'package:llama/bindings.dart';
import 'package:llama/llama.dart';

void main() {
  runApp(const LlamaApp());
}

class LlamaApp extends StatefulWidget {
  const LlamaApp({super.key});

  @override
  State<LlamaApp> createState() => _LlamaAppState();
}

class _LlamaAppState extends State<LlamaApp> {
  final TextEditingController _controller = TextEditingController();
  final List<ChatMessage> _messages = [];
  String? _model;

  void _loadModel() async {
    final result = await FilePicker.platform.pickFiles(
      dialogTitle: "Load Model File",
      type: FileType.any,
      allowMultiple: false,
      allowCompression: false
    );

    if (result != null && result.files.isNotEmpty) {
      setState(() {
        _model = result.files.single.path!;
      });
    }
  }

  void _onSubmitted(String value) async {
    if (_model == null) {
      return;
    }

    setState(() {
      _messages.add(ChatMessage(role: 'user', content: value));
      _controller.clear();
    });

    final llamaCpp = LlamaCPP(
      _model!,
      ModelParams(),
      ContextParams(),
      SamplingParams(
        minP: PArguments(p: 0.05, minKeep: 1),
        temperature: TemperatureArguments(temperature: 0.8),
        seed: Random().nextInt(1000000)
      )
    );

    print('LlamaCPP created');

    Stream<String> stream = llamaCpp.prompt(_messages);

    setState(() {
      _messages.add(ChatMessage(role: 'assistant', content: ""));
    });

    stream.listen((message) {
      setState(() {
        final newContent = _messages.last.content + message;
        final newLastMessage = ChatMessage(role: 'assistant', content: newContent);
        _messages[_messages.length - 1] = newLastMessage;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: buildHome()
    );
  }

  Widget buildHome() {
    return Scaffold(
      appBar: buildAppBar(),
      body: buildBody(),
    );
  }

  PreferredSizeWidget buildAppBar() {
    return AppBar(
      title: Text(_model ?? 'No model loaded'),
      leading: IconButton(
        icon: const Icon(Icons.folder_open),
        onPressed: _loadModel,
      ),
    );
  }

  Widget buildBody() {
    return Column(
      children: [
        Expanded(
          child: ListView.builder(
            itemCount: _messages.length,
            itemBuilder: (context, index) {
              final message = _messages[index];
              return ListTile(
                title: Text(message.role),
                subtitle: Text(message.content),
              );
            },
          ),
        ),
        buildInputField(),
      ],
    );
  }

  Widget buildInputField() {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _controller,
              onSubmitted: _onSubmitted,
              decoration: const InputDecoration(
                labelText: 'Enter your message',
                border: OutlineInputBorder(),
              ),
            ),
          ),
          IconButton(
            icon: const Icon(Icons.send),
            onPressed: () {
              _onSubmitted(_controller.text);
            },
          ),
        ],
      ),
    );
  }
}