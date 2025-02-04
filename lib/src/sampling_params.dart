part of '../llama.dart';

typedef PArgs = ({
  double p, 
  int minKeep
});

typedef TemperatureArgs = ({
  double temperature, 
  double? delta, 
  double? exponent
});

typedef XtcArgs = ({
  double p, 
  double t, 
  int minKeep, 
  int seed
});

typedef MirostatArgs = ({
  int nVocab, 
  int seed, 
  double tau, 
  double eta, 
  int m
});

typedef MirostatV2Args = ({
  int seed, 
  double tau, 
  double eta
});

typedef GrammarArgs = ({
  String str, 
  String root
});

typedef PenaltiesArgs = ({
  int lastN, 
  double repeat, 
  double frequency, 
  double present
});

typedef DrySamplerArgs = ({
  int nCtxTrain, 
  double multiplier, 
  double dryBase, 
  int allowedLength, 
  int penaltyLastN, 
  List<String> sequenceBreakers
});

class SamplingParams {
  bool greedy;
  bool infill;
  int? seed;
  int? topK;
  PArgs? topP;
  PArgs? minP;
  PArgs? typicalP;
  TemperatureArgs? temperature;
  XtcArgs? xtc;
  MirostatArgs? mirostat;
  MirostatV2Args? mirostatV2;
  GrammarArgs? grammar;
  PenaltiesArgs? penalties;
  DrySamplerArgs? drySampler;

  SamplingParams({
    this.greedy = false,
    this.infill = false,
    this.seed,
    this.topK,
    this.topP,
    this.minP,
    this.typicalP,
    this.temperature,
    this.xtc,
    this.mirostat,
    this.mirostatV2,
    this.grammar,
    this.penalties,
    this.drySampler
  });

  factory SamplingParams.fromBytes(List<int> bytes) {
    final buffer = Uint8List.fromList(bytes);
    final byteData = ByteData.sublistView(buffer);
    int offset = 0;
    // Read the always‑present booleans.
    bool greedy = buffer[offset] != 0;
    offset += 1;
    bool infill = buffer[offset] != 0;
    offset += 1;
    // Read the 4‑byte header.
    int header = byteData.getUint32(offset, Endian.little);
    offset += 4;

    int? seed;
    int? topK;
    PArgs? topP;
    PArgs? minP;
    PArgs? typicalP;
    TemperatureArgs? temperature;
    XtcArgs? xtc;
    MirostatArgs? mirostat;
    MirostatV2Args? mirostatV2;
    GrammarArgs? grammar;
    PenaltiesArgs? penalties;
    DrySamplerArgs? drySampler;

    if ((header & (1 << 0)) != 0) {
      seed = byteData.getInt64(offset, Endian.little);
      offset += 8;
    }
    if ((header & (1 << 1)) != 0) {
      topK = byteData.getInt64(offset, Endian.little);
      offset += 8;
    }
    if ((header & (1 << 2)) != 0) {
      double p = byteData.getFloat64(offset, Endian.little);
      offset += 8;
      int minKeep = byteData.getInt64(offset, Endian.little);
      offset += 8;
      topP = (p: p, minKeep: minKeep);
    }
    if ((header & (1 << 3)) != 0) {
      double p = byteData.getFloat64(offset, Endian.little);
      offset += 8;
      int minKeep = byteData.getInt64(offset, Endian.little);
      offset += 8;
      minP = (p: p, minKeep: minKeep);
    }
    if ((header & (1 << 4)) != 0) {
      double p = byteData.getFloat64(offset, Endian.little);
      offset += 8;
      int minKeep = byteData.getInt64(offset, Endian.little);
      offset += 8;
      typicalP = (p: p, minKeep: minKeep);
    }
    if ((header & (1 << 5)) != 0) {
      double temp = byteData.getFloat64(offset, Endian.little);
      offset += 8;
      int tempHeader = buffer[offset];
      offset += 1;
      double? delta;
      double? exponent;
      if ((tempHeader & (1 << 0)) != 0) {
        delta = byteData.getFloat64(offset, Endian.little);
        offset += 8;
      }
      if ((tempHeader & (1 << 1)) != 0) {
        exponent = byteData.getFloat64(offset, Endian.little);
        offset += 8;
      }
      temperature = (temperature: temp, delta: delta, exponent: exponent);
    }
    if ((header & (1 << 6)) != 0) {
      double p = byteData.getFloat64(offset, Endian.little);
      offset += 8;
      double t = byteData.getFloat64(offset, Endian.little);
      offset += 8;
      int minKeep = byteData.getInt64(offset, Endian.little);
      offset += 8;
      int seedXtc = byteData.getInt64(offset, Endian.little);
      offset += 8;
      xtc = (p: p, t: t, minKeep: minKeep, seed: seedXtc);
    }
    if ((header & (1 << 7)) != 0) {
      int nVocab = byteData.getInt64(offset, Endian.little);
      offset += 8;
      int seedMiro = byteData.getInt64(offset, Endian.little);
      offset += 8;
      double tau = byteData.getFloat64(offset, Endian.little);
      offset += 8;
      double eta = byteData.getFloat64(offset, Endian.little);
      offset += 8;
      int m = byteData.getInt64(offset, Endian.little);
      offset += 8;
      mirostat = (nVocab: nVocab, seed: seedMiro, tau: tau, eta: eta, m: m);
    }
    if ((header & (1 << 8)) != 0) {
      int seedV2 = byteData.getInt64(offset, Endian.little);
      offset += 8;
      double tau = byteData.getFloat64(offset, Endian.little);
      offset += 8;
      double eta = byteData.getFloat64(offset, Endian.little);
      offset += 8;
      mirostatV2 = (seed: seedV2, tau: tau, eta: eta);
    }
    if ((header & (1 << 9)) != 0) {
      int strLen = byteData.getUint32(offset, Endian.little);
      offset += 4;
      String str = utf8.decode(buffer.sublist(offset, offset + strLen));
      offset += strLen;
      int rootLen = byteData.getUint32(offset, Endian.little);
      offset += 4;
      String root = utf8.decode(buffer.sublist(offset, offset + rootLen));
      offset += rootLen;
      grammar = (str: str, root: root);
    }
    if ((header & (1 << 10)) != 0) {
      int lastN = byteData.getInt64(offset, Endian.little);
      offset += 8;
      double repeat = byteData.getFloat64(offset, Endian.little);
      offset += 8;
      double frequency = byteData.getFloat64(offset, Endian.little);
      offset += 8;
      double present = byteData.getFloat64(offset, Endian.little);
      offset += 8;
      penalties = (lastN: lastN, repeat: repeat, frequency: frequency, present: present);
    }
    if ((header & (1 << 11)) != 0) {
      int nCtxTrain = byteData.getInt64(offset, Endian.little);
      offset += 8;
      double multiplier = byteData.getFloat64(offset, Endian.little);
      offset += 8;
      double dryBase = byteData.getFloat64(offset, Endian.little);
      offset += 8;
      int allowedLength = byteData.getInt64(offset, Endian.little);
      offset += 8;
      int penaltyLastN = byteData.getInt64(offset, Endian.little);
      offset += 8;
      int listLength = byteData.getUint32(offset, Endian.little);
      offset += 4;
      List<String> sequenceBreakers = [];
      for (int i = 0; i < listLength; i++) {
        int sLen = byteData.getUint32(offset, Endian.little);
        offset += 4;
        String s = utf8.decode(buffer.sublist(offset, offset + sLen));
        offset += sLen;
        sequenceBreakers.add(s);
      }
      drySampler = (nCtxTrain: nCtxTrain,
                    multiplier: multiplier,
                    dryBase: dryBase,
                    allowedLength: allowedLength,
                    penaltyLastN: penaltyLastN,
                    sequenceBreakers: sequenceBreakers);
    }
    return SamplingParams(
      greedy: greedy,
      infill: infill,
      seed: seed,
      topK: topK,
      topP: topP,
      minP: minP,
      typicalP: typicalP,
      temperature: temperature,
      xtc: xtc,
      mirostat: mirostat,
      mirostatV2: mirostatV2,
      grammar: grammar,
      penalties: penalties,
      drySampler: drySampler,
    );
  }

  ffi.Pointer<llama_sampler> toNative(ffi.Pointer<llama_vocab> vocab) {
    final sampler = LlamaCppNative.lib.llama_sampler_chain_init(LlamaCppNative.lib.llama_sampler_chain_default_params());

    if (greedy) {
      LlamaCppNative.lib.llama_sampler_chain_add(sampler, LlamaCppNative.lib.llama_sampler_init_greedy());
    }

    if (infill) {
      LlamaCppNative.lib.llama_sampler_chain_add(sampler, LlamaCppNative.lib.llama_sampler_init_infill(vocab));
    }

    if (seed != null) {
      LlamaCppNative.lib.llama_sampler_chain_add(sampler, LlamaCppNative.lib.llama_sampler_init_dist(seed!));
    }

    if (topK != null) {
      LlamaCppNative.lib.llama_sampler_chain_add(sampler, LlamaCppNative.lib.llama_sampler_init_top_k(topK!));
    }

    if (topP != null) {
      LlamaCppNative.lib.llama_sampler_chain_add(sampler, LlamaCppNative.lib.llama_sampler_init_top_p(topP!.p, topP!.minKeep));
    }

    if (minP != null) {
      LlamaCppNative.lib.llama_sampler_chain_add(sampler, LlamaCppNative.lib.llama_sampler_init_min_p(minP!.p, minP!.minKeep));
    }

    if (typicalP != null) {
      LlamaCppNative.lib.llama_sampler_chain_add(sampler, LlamaCppNative.lib.llama_sampler_init_typical(typicalP!.p, typicalP!.minKeep));
    }

    if (temperature != null) {
      if (temperature!.delta == null && temperature!.exponent == null) {
        LlamaCppNative.lib.llama_sampler_chain_add(
          sampler, 
          LlamaCppNative.lib.llama_sampler_init_temp(temperature!.temperature)
        );
      } 
      else {
        LlamaCppNative.lib.llama_sampler_chain_add(
          sampler, 
          LlamaCppNative.lib.llama_sampler_init_temp_ext(temperature!.temperature, temperature!.delta!, temperature!.exponent!)
        );
      }
    }

    if (xtc != null) {
      LlamaCppNative.lib.llama_sampler_chain_add(
        sampler, 
        LlamaCppNative.lib.llama_sampler_init_xtc(xtc!.p, xtc!.t, xtc!.minKeep, xtc!.seed)
      );
    }

    if (mirostat != null) {
      LlamaCppNative.lib.llama_sampler_chain_add(
        sampler, 
        LlamaCppNative.lib.llama_sampler_init_mirostat(mirostat!.nVocab, mirostat!.seed, mirostat!.tau, mirostat!.eta, mirostat!.m)
      );
    }

    if (mirostatV2 != null) {
      LlamaCppNative.lib.llama_sampler_chain_add(
        sampler, 
        LlamaCppNative.lib.llama_sampler_init_mirostat_v2(mirostatV2!.seed, mirostatV2!.tau, mirostatV2!.eta)
      );
    }

    if (grammar != null) {
      LlamaCppNative.lib.llama_sampler_chain_add(
        sampler, 
        LlamaCppNative.lib.llama_sampler_init_grammar(
          vocab, 
          grammar!.str.toNativeUtf8().cast<ffi.Char>(), 
          grammar!.root.toNativeUtf8().cast<ffi.Char>()
        )
      );
    }

    if (penalties != null) {
      LlamaCppNative.lib.llama_sampler_chain_add(
        sampler, 
        LlamaCppNative.lib.llama_sampler_init_penalties(
          penalties!.lastN, 
          penalties!.repeat, 
          penalties!.frequency, 
          penalties!.present
        )
      );
    }

    if (drySampler != null) {
      final sequenceBreakers = calloc<ffi.Pointer<ffi.Char>>(drySampler!.sequenceBreakers.length);
      for (var i = 0; i < drySampler!.sequenceBreakers.length; i++) {
        sequenceBreakers[i] = drySampler!.sequenceBreakers[i].toNativeUtf8().cast<ffi.Char>();
      }

      LlamaCppNative.lib.llama_sampler_chain_add(
        sampler, 
        LlamaCppNative.lib.llama_sampler_init_dry(
          vocab,
          drySampler!.nCtxTrain, 
          drySampler!.multiplier, 
          drySampler!.dryBase, 
          drySampler!.allowedLength, 
          drySampler!.penaltyLastN, 
          sequenceBreakers,
          drySampler!.sequenceBreakers.length
        )
      );
    }

    return sampler;
  }

  /// Converts this [SamplingParams] instance into a binary representation as a
  /// [List<int>] (a byte array). The encoding is as follows:
  ///
  /// 1. Write two bytes for the always‑present booleans: `greedy` and `infill`.
  /// 2. Write a 4‑byte header (a bitmask for the 12 optional fields, in the order below):
  ///    - Bit 0: `seed`
  ///    - Bit 1: `topK`
  ///    - Bit 2: `topP`
  ///    - Bit 3: `minP`
  ///    - Bit 4: `typicalP`
  ///    - Bit 5: `temperature`
  ///    - Bit 6: `xtc`
  ///    - Bit 7: `mirostat`
  ///    - Bit 8: `mirostatV2`
  ///    - Bit 9: `grammar`
  ///    - Bit 10: `penalties`
  ///    - Bit 11: `drySampler`
  /// 3. For each optional field that is present, write its data in a fixed order.
  ///
  /// For record types:
  /// - A `PArgs` record is written as:
  ///   - double `p` (8 bytes)
  ///   - int `minKeep` (8 bytes)
  /// - A `TemperatureArgs` record is written as:
  ///   - double `temperature` (8 bytes)
  ///   - 1‑byte sub‑header (bit 0 indicates whether `delta` is present, bit 1 for `exponent`)
  ///   - then (if present) double `delta` (8 bytes) and/or double `exponent` (8 bytes)
  /// - A `XtcArgs` record is written as: double, double, int, int (8+8+8+8 bytes)
  /// - A `MirostatArgs` record is written as: int, int, double, double, int (8×5 bytes)
  /// - A `MirostatV2Args` record is written as: int, double, double (8+8+8 bytes)
  /// - A `GrammarArgs` record is written as:
  ///   - int32 string-length for `str` + UTF‑8 bytes for `str`
  ///   - int32 string-length for `root` + UTF‑8 bytes for `root`
  /// - A `PenaltiesArgs` record is written as: int, double, double, double (8×4 bytes)
  /// - A `DrySamplerArgs` record is written as:
  ///   - int, double, double, int, int (40 bytes total)
  ///   - int32 number-of-sequenceBreakers,
  ///   - then for each string: int32 string-length + UTF‑8 bytes.
  List<int> toBytes() {
    // Base length: 2 bytes for booleans + 4 bytes for header.
    int length = 6;
    int header = 0;
    // We reserve 12 bits (one for each optional field) in the header.
    // Optional fields in order:
    // 0: seed, 1: topK, 2: topP, 3: minP, 4: typicalP, 5: temperature,
    // 6: xtc, 7: mirostat, 8: mirostatV2, 9: grammar, 10: penalties, 11: drySampler
    if (seed != null)      { header |= (1 << 0); length += 8; }
    if (topK != null)      { header |= (1 << 1); length += 8; }
    if (topP != null)      { header |= (1 << 2); length += 8 + 8; }
    if (minP != null)      { header |= (1 << 3); length += 8 + 8; }
    if (typicalP != null)  { header |= (1 << 4); length += 8 + 8; }
    if (temperature != null) {
      header |= (1 << 5);
      // TemperatureArgs always writes the required temperature (8 bytes)
      // plus a 1-byte sub‑header for the optional fields.
      int tempLength = 8 + 1;
      if (temperature!.delta != null)    tempLength += 8;
      if (temperature!.exponent != null) tempLength += 8;
      length += tempLength;
    }
    if (xtc != null)         { header |= (1 << 6); length += 8 + 8 + 8 + 8; } // 32 bytes
    if (mirostat != null)    { header |= (1 << 7); length += 8 + 8 + 8 + 8 + 8; } // 40 bytes
    if (mirostatV2 != null)  { header |= (1 << 8); length += 8 + 8 + 8; } // 24 bytes
    if (grammar != null) {
      header |= (1 << 9);
      // For each string, 4 bytes for length + bytes for the UTF-8 encoding.
      List<int> strBytes = utf8.encode(grammar!.str);
      List<int> rootBytes = utf8.encode(grammar!.root);
      length += 4 + strBytes.length + 4 + rootBytes.length;
    }
    if (penalties != null)   { header |= (1 << 10); length += 8 + 8 + 8 + 8; } // 32 bytes
    if (drySampler != null) {
      header |= (1 << 11);
      // Fixed part: 8×5 = 40 bytes, plus the list of sequenceBreakers.
      int dsLength = 40 + 4; // 4 bytes to store the list length.
      for (final s in drySampler!.sequenceBreakers) {
        List<int> sBytes = utf8.encode(s);
        dsLength += 4 + sBytes.length;
      }
      length += dsLength;
    }

    final buffer = Uint8List(length);
    final byteData = ByteData.sublistView(buffer);
    int offset = 0;
    // Write the two booleans (1 byte each).
    buffer[offset] = greedy ? 1 : 0;
    offset += 1;
    buffer[offset] = infill ? 1 : 0;
    offset += 1;
    // Write the 4-byte header.
    byteData.setUint32(offset, header, Endian.little);
    offset += 4;

    // For each optional field (in the fixed order), write its data if present.
    if (seed != null) {
      byteData.setInt64(offset, seed!, Endian.little);
      offset += 8;
    }
    if (topK != null) {
      byteData.setInt64(offset, topK!, Endian.little);
      offset += 8;
    }
    if (topP != null) {
      // Write PArgs: double p then int64 minKeep.
      byteData.setFloat64(offset, topP!.p, Endian.little);
      offset += 8;
      byteData.setInt64(offset, topP!.minKeep, Endian.little);
      offset += 8;
    }
    if (minP != null) {
      byteData.setFloat64(offset, minP!.p, Endian.little);
      offset += 8;
      byteData.setInt64(offset, minP!.minKeep, Endian.little);
      offset += 8;
    }
    if (typicalP != null) {
      byteData.setFloat64(offset, typicalP!.p, Endian.little);
      offset += 8;
      byteData.setInt64(offset, typicalP!.minKeep, Endian.little);
      offset += 8;
    }
    if (temperature != null) {
      // Write the required field.
      byteData.setFloat64(offset, temperature!.temperature, Endian.little);
      offset += 8;
      // Write a 1-byte sub-header for the optional fields (delta and exponent).
      int tempHeader = 0;
      if (temperature!.delta != null)    tempHeader |= 1 << 0;
      if (temperature!.exponent != null) tempHeader |= 1 << 1;
      buffer[offset] = tempHeader;
      offset += 1;
      if (temperature!.delta != null) {
        byteData.setFloat64(offset, temperature!.delta!, Endian.little);
        offset += 8;
      }
      if (temperature!.exponent != null) {
        byteData.setFloat64(offset, temperature!.exponent!, Endian.little);
        offset += 8;
      }
    }
    if (xtc != null) {
      // XtcArgs: double p, double t, int64 minKeep, int64 seed.
      byteData.setFloat64(offset, xtc!.p, Endian.little);
      offset += 8;
      byteData.setFloat64(offset, xtc!.t, Endian.little);
      offset += 8;
      byteData.setInt64(offset, xtc!.minKeep, Endian.little);
      offset += 8;
      byteData.setInt64(offset, xtc!.seed, Endian.little);
      offset += 8;
    }
    if (mirostat != null) {
      // MirostatArgs: int64 nVocab, int64 seed, double tau, double eta, int64 m.
      byteData.setInt64(offset, mirostat!.nVocab, Endian.little);
      offset += 8;
      byteData.setInt64(offset, mirostat!.seed, Endian.little);
      offset += 8;
      byteData.setFloat64(offset, mirostat!.tau, Endian.little);
      offset += 8;
      byteData.setFloat64(offset, mirostat!.eta, Endian.little);
      offset += 8;
      byteData.setInt64(offset, mirostat!.m, Endian.little);
      offset += 8;
    }
    if (mirostatV2 != null) {
      // MirostatV2Args: int64 seed, double tau, double eta.
      byteData.setInt64(offset, mirostatV2!.seed, Endian.little);
      offset += 8;
      byteData.setFloat64(offset, mirostatV2!.tau, Endian.little);
      offset += 8;
      byteData.setFloat64(offset, mirostatV2!.eta, Endian.little);
      offset += 8;
    }
    if (grammar != null) {
      // GrammarArgs: first the string for `str`, then for `root`.
      List<int> strBytes = utf8.encode(grammar!.str);
      List<int> rootBytes = utf8.encode(grammar!.root);
      byteData.setUint32(offset, strBytes.length, Endian.little);
      offset += 4;
      buffer.setRange(offset, offset + strBytes.length, strBytes);
      offset += strBytes.length;
      byteData.setUint32(offset, rootBytes.length, Endian.little);
      offset += 4;
      buffer.setRange(offset, offset + rootBytes.length, rootBytes);
      offset += rootBytes.length;
    }
    if (penalties != null) {
      // PenaltiesArgs: int64 lastN, double repeat, double frequency, double present.
      byteData.setInt64(offset, penalties!.lastN, Endian.little);
      offset += 8;
      byteData.setFloat64(offset, penalties!.repeat, Endian.little);
      offset += 8;
      byteData.setFloat64(offset, penalties!.frequency, Endian.little);
      offset += 8;
      byteData.setFloat64(offset, penalties!.present, Endian.little);
      offset += 8;
    }
    if (drySampler != null) {
      // DrySamplerArgs: int64 nCtxTrain, double multiplier, double dryBase,
      // int64 allowedLength, int64 penaltyLastN.
      byteData.setInt64(offset, drySampler!.nCtxTrain, Endian.little);
      offset += 8;
      byteData.setFloat64(offset, drySampler!.multiplier, Endian.little);
      offset += 8;
      byteData.setFloat64(offset, drySampler!.dryBase, Endian.little);
      offset += 8;
      byteData.setInt64(offset, drySampler!.allowedLength, Endian.little);
      offset += 8;
      byteData.setInt64(offset, drySampler!.penaltyLastN, Endian.little);
      offset += 8;
      // Now write the list of sequenceBreakers:
      byteData.setUint32(offset, drySampler!.sequenceBreakers.length, Endian.little);
      offset += 4;
      for (final s in drySampler!.sequenceBreakers) {
        List<int> sBytes = utf8.encode(s);
        byteData.setUint32(offset, sBytes.length, Endian.little);
        offset += 4;
        buffer.setRange(offset, offset + sBytes.length, sBytes);
        offset += sBytes.length;
      }
    }
    return buffer;
  }
}