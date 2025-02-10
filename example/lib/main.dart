import 'dart:io';

import 'package:flutter/material.dart';

import 'package:file_picker/file_picker.dart';
import 'package:lcpp/lcpp.dart';

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
  Llama? _model;
  String? _modelPath;

  void _loadModel() async {
    final result = await FilePicker.platform.pickFiles(
        dialogTitle: "Load Model File",
        type: FileType.any,
        allowMultiple: false,
        allowCompression: false);

    if (result == null ||
        result.files.isEmpty ||
        result.files.single.path == null) {
      throw Exception('No file selected');
    }

    File resultFile = File(result.files.single.path!);

    final exists = await resultFile.exists();
    if (!exists) {
      throw Exception('File does not exist');
    }

    final llamaCpp = LlamaIsolated(
        modelParams: ModelParams(path: result.files.single.path!),
        contextParams: const ContextParams(nCtx: 2048, nBatch: 2048),
        samplingParams: const SamplingParams(greedy: true));

    setState(() {
      _model = llamaCpp;
      _modelPath = result.files.single.path;
    });
  }

  void _onSubmitted(String value) async {
    if (_model == null) {
      return;
    }

    setState(() {
      _messages.add(UserChatMessage(value));
      _controller.clear();
    });

    final stream = _model!.prompt(_messages.copy());

    _messages.add(AssistantChatMessage(''));

    await for (var response in stream) {
      setState(() {
        _messages.last.content += response;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(home: buildHome());
  }

  Widget buildHome() {
    return Scaffold(
      appBar: buildAppBar(),
      body: buildBody(),
    );
  }

  PreferredSizeWidget buildAppBar() {
    return AppBar(
        title: Text(_modelPath ?? 'No model loaded'),
        leading: IconButton(
          icon: const Icon(Icons.folder_open),
          onPressed: _loadModel,
        ));
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
