# lcpp

lcpp is a dart implementation of llama.cpp used by the mobile artificial intelligence distribution (maid)

## Features

- Optional use of isolations for parallel processing
- Streaming support

## Getting started

To use this package, add `lcpp` as a [dependency in your pubspec.yaml file](https://flutter.io/platform-plugins/).

```yaml
dependencies:
  lcpp: ^0.0.8
```

then you can import llama in your Dart code

```dart
import 'package:lcpp/lcpp.dart';
```

## Usage

Below is a simple example of how to use llama

```dart
final llama = Llama(
  modelParams: ModelParams(
    path: 'path/to/model',
  ),
  contextParams: const ContextParams(
    nCtx: 2048,
    nBatch: 2048
  ),
  samplingParams: const SamplingParams(
    greedy: true
  ),
  isolate: true
);

final messages = [
  ChatMessage.withRole(role: 'user', content: 'Hello World'),
];

final stream = llama.prompt(messages);

stream.listen((event) {
  print(event);
});
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
```
MIT License

Copyright (c) 2023 Dane Madsen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```