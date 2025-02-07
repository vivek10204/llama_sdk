part of '../llama.dart';

class TopPArguments {
  final double p;
  final int minKeep;

  const TopPArguments({
    required this.p,
    required this.minKeep
  });

  factory TopPArguments.fromMap(Map<String, dynamic> map) => TopPArguments(
    p: map['p'],
    minKeep: map['minKeep']
  );

  Map<String, dynamic> toMap() => {
    'p': p,
    'minKeep': minKeep
  };
  
  void add(ffi.Pointer<llama_sampler> sampler) {
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_top_p(p, minKeep));
  }
}

class MinPArguments {
  final double p;
  final int minKeep;

  const MinPArguments({
    required this.p,
    required this.minKeep
  });

  factory MinPArguments.fromMap(Map<String, dynamic> map) => MinPArguments(
    p: map['p'],
    minKeep: map['minKeep']
  );

  Map<String, dynamic> toMap() => {
    'p': p,
    'minKeep': minKeep
  };
  
  void add(ffi.Pointer<llama_sampler> sampler) {
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_min_p(p, minKeep));
  }
}

class TypicalPArguments {
  final double p;
  final int minKeep;

  const TypicalPArguments({
    required this.p,
    required this.minKeep
  });

  factory TypicalPArguments.fromMap(Map<String, dynamic> map) => TypicalPArguments(
    p: map['p'],
    minKeep: map['minKeep']
  );

  Map<String, dynamic> toMap() => {
    'p': p,
    'minKeep': minKeep
  };
  
  void add(ffi.Pointer<llama_sampler> sampler) {
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_typical(p, minKeep));
  }
}

class TemperatureArguments {
  final double temperature;
  final double? delta;
  final double? exponent;

  const TemperatureArguments({
    required this.temperature,
    this.delta,
    this.exponent
  });

  factory TemperatureArguments.fromMap(Map<String, dynamic> map) => TemperatureArguments(
    temperature: map['temperature'],
    delta: map['delta'],
    exponent: map['exponent']
  );

  Map<String, dynamic> toMap() => {
    'temperature': temperature,
    'delta': delta,
    'exponent': exponent
  };

  void add(ffi.Pointer<llama_sampler> sampler) {
    if (delta == null || exponent == null) {
      lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_temp(temperature));
    } 
    else {
      lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_temp_ext(temperature, delta!, exponent!));
    }
  }
}

class XtcArguments {
  final double p;
  final double t;
  final int minKeep;
  final int seed;

  const XtcArguments({
    required this.p,
    required this.t,
    required this.minKeep,
    required this.seed
  });

  factory XtcArguments.fromMap(Map<String, dynamic> map) => XtcArguments(
    p: map['p'],
    t: map['t'],
    minKeep: map['minKeep'],
    seed: map['seed']
  );

  Map<String, dynamic> toMap() => {
    'p': p,
    't': t,
    'minKeep': minKeep,
    'seed': seed
  };

  void add(ffi.Pointer<llama_sampler> sampler) {
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_xtc(p, t, minKeep, seed));
  }
}

class MirostatArguments {
  final int nVocab;
  final int seed;
  final double tau;
  final double eta;
  final int m;

  const MirostatArguments({
    required this.nVocab,
    required this.seed,
    required this.tau,
    required this.eta,
    required this.m
  });

  factory MirostatArguments.fromMap(Map<String, dynamic> map) => MirostatArguments(
    nVocab: map['nVocab'],
    seed: map['seed'],
    tau: map['tau'],
    eta: map['eta'],
    m: map['m']
  );

  Map<String, dynamic> toMap() => {
    'nVocab': nVocab,
    'seed': seed,
    'tau': tau,
    'eta': eta,
    'm': m
  };

  void add(ffi.Pointer<llama_sampler> sampler) {
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_mirostat(nVocab, seed, tau, eta, m));
  }
}

class MirostatV2Arguments {
  final int seed;
  final double tau;
  final double eta;

  const MirostatV2Arguments({
    required this.seed,
    required this.tau,
    required this.eta
  });

  factory MirostatV2Arguments.fromMap(Map<String, dynamic> map) => MirostatV2Arguments(
    seed: map['seed'],
    tau: map['tau'],
    eta: map['eta']
  );

  Map<String, dynamic> toMap() => {
    'seed': seed,
    'tau': tau,
    'eta': eta
  };

  void add(ffi.Pointer<llama_sampler> sampler) {
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_mirostat_v2(seed, tau, eta));
  }
}

class GrammarArguments {
  final String str;
  final String root;

  const GrammarArguments({
    required this.str,
    required this.root
  });

  factory GrammarArguments.fromMap(Map<String, dynamic> map) => GrammarArguments(
    str: map['str'],
    root: map['root']
  );

  Map<String, dynamic> toMap() => {
    'str': str,
    'root': root
  };

  void add(ffi.Pointer<llama_sampler> sampler, ffi.Pointer<llama_vocab> vocab) {
    lib.llama_sampler_chain_add(
      sampler, 
      lib.llama_sampler_init_grammar(
        vocab, 
        str.toNativeUtf8().cast<ffi.Char>(), 
        root.toNativeUtf8().cast<ffi.Char>()
      )
    );
  }
}

class PenaltiesArguments {
  final int lastN;
  final double repeat;
  final double frequency;
  final double present;

  const PenaltiesArguments({
    required this.lastN,
    required this.repeat,
    required this.frequency,
    required this.present
  });

  factory PenaltiesArguments.fromMap(Map<String, dynamic> map) => PenaltiesArguments(
    lastN: map['lastN'],
    repeat: map['repeat'],
    frequency: map['frequency'],
    present: map['present']
  );

  Map<String, dynamic> toMap() => {
    'lastN': lastN,
    'repeat': repeat,
    'frequency': frequency,
    'present': present
  };

  void add(ffi.Pointer<llama_sampler> sampler) {
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_penalties(lastN, repeat, frequency, present));
  }
}

class DrySamplerArguments {
  final int nCtxTrain;
  final double multiplier;
  final double dryBase;
  final int allowedLength;
  final int penaltyLastN;
  final List<String> sequenceBreakers;

  const DrySamplerArguments({
    required this.nCtxTrain,
    required this.multiplier,
    required this.dryBase,
    required this.allowedLength,
    required this.penaltyLastN,
    required this.sequenceBreakers
  });

  factory DrySamplerArguments.fromMap(Map<String, dynamic> map) => DrySamplerArguments(
    nCtxTrain: map['nCtxTrain'],
    multiplier: map['multiplier'],
    dryBase: map['dryBase'],
    allowedLength: map['allowedLength'],
    penaltyLastN: map['penaltyLastN'],
    sequenceBreakers: List<String>.from(map['sequenceBreakers'])
  );

  Map<String, dynamic> toMap() => {
    'nCtxTrain': nCtxTrain,
    'multiplier': multiplier,
    'dryBase': dryBase,
    'allowedLength': allowedLength,
    'penaltyLastN': penaltyLastN,
    'sequenceBreakers': sequenceBreakers
  };

  void add(ffi.Pointer<llama_sampler> sampler, ffi.Pointer<llama_vocab> vocab) {
    final sequenceBreakers = calloc<ffi.Pointer<ffi.Char>>(this.sequenceBreakers.length);
    for (var i = 0; i < this.sequenceBreakers.length; i++) {
      sequenceBreakers[i] = this.sequenceBreakers[i].toNativeUtf8().cast<ffi.Char>();
    }

    lib.llama_sampler_chain_add(
      sampler, 
      lib.llama_sampler_init_dry(
        vocab,
        nCtxTrain, 
        multiplier, 
        dryBase, 
        allowedLength, 
        penaltyLastN, 
        sequenceBreakers,
        this.sequenceBreakers.length
      )
    );
  }
}

class SamplingParams {
  final bool greedy;
  final bool infill;
  final int? seed;
  final int? topK;
  final TopPArguments? topP;
  final MinPArguments? minP;
  final TypicalPArguments? typicalP;
  final TemperatureArguments? temperature;
  final XtcArguments? xtc;
  final MirostatArguments? mirostat;
  final MirostatV2Arguments? mirostatV2;
  final GrammarArguments? grammar;
  final PenaltiesArguments? penalties;
  final DrySamplerArguments? drySampler;

  const SamplingParams({
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

  factory SamplingParams.fromMap(Map<String, dynamic> map) => SamplingParams(
    greedy: map['greedy'],
    infill: map['infill'],
    seed: map['seed'],
    topK: map['topK'],
    topP: map['topP'] != null ? TopPArguments.fromMap(map['topP']) : null,
    minP: map['minP'] != null ? MinPArguments.fromMap(map['minP']) : null,
    typicalP: map['typicalP'] != null ? TypicalPArguments.fromMap(map['typicalP']) : null,
    temperature: map['temperature'] != null ? TemperatureArguments.fromMap(map['temperature']) : null,
    xtc: map['xtc'] != null ? XtcArguments.fromMap(map['xtc']) : null,
    mirostat: map['mirostat'] != null ? MirostatArguments.fromMap(map['mirostat']) : null,
    mirostatV2: map['mirostatV2'] != null ? MirostatV2Arguments.fromMap(map['mirostatV2']) : null,
    grammar: map['grammar'] != null ? GrammarArguments.fromMap(map['grammar']) : null,
    penalties: map['penalties'] != null ? PenaltiesArguments.fromMap(map['penalties']) : null,
    drySampler: map['drySampler'] != null ? DrySamplerArguments.fromMap(map['drySampler']) : null
  );

  factory SamplingParams.fromJson(String source) => SamplingParams.fromMap(jsonDecode(source));

  ffi.Pointer<llama_sampler> toNative(ffi.Pointer<llama_vocab> vocab) {
    final sampler = lib.llama_sampler_chain_init(lib.llama_sampler_chain_default_params());

    if (greedy) {
      lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_greedy());
    }

    if (infill) {
      lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_infill(vocab));
    }

    if (seed != null) {
      lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_dist(seed!));
    }

    if (topK != null) {
      lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_top_k(topK!));
    }

    topP?.add(sampler);

    minP?.add(sampler);

    typicalP?.add(sampler);

    temperature?.add(sampler);

    xtc?.add(sampler);

    mirostat?.add(sampler);

    mirostatV2?.add(sampler);

    grammar?.add(sampler, vocab);

    penalties?.add(sampler);

    drySampler?.add(sampler, vocab);

    return sampler;
  }

  Map<String, dynamic> toMap() => {
    'greedy': greedy,
    'infill': infill,
    'seed': seed,
    'topK': topK,
    'topP': topP?.toMap(),
    'minP': minP?.toMap(),
    'typicalP': typicalP?.toMap(),
    'temperature': temperature?.toMap(),
    'xtc': xtc?.toMap(),
    'mirostat': mirostat?.toMap(),
    'mirostatV2': mirostatV2?.toMap(),
    'grammar': grammar?.toMap(),
    'penalties': penalties?.toMap(),
    'drySampler': drySampler?.toMap()
  };

  String toJson() => jsonEncode(toMap());
}