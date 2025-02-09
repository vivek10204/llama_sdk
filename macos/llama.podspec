#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run pod lib lint llama.podspec to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'llama'
  s.version          = '0.0.1'
  s.summary          = 'A Flutter FFI plugin for interfacing with llama_cpp.'
  s.description      = <<-DESC
  A new Flutter FFI plugin project.
                       DESC
  s.homepage         = 'https://github.com/Mobile-Artificial-Intelligence/llama_dart'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Dane Madsen' => 'dane_madsen@hotmail.com' }
  s.dependency 'FlutterMacOS'
  s.swift_version = '5.0'
  s.source           = { :path => '.' }
  s.prepare_command = <<-CMD
    set -e
    set -u
    set -o pipefail

    # Run CMake to configure and build
    cd ../src/llama_cpp
    cmake -B build
    cmake --build build --config Release

    cp ./build/bin/*.dylib ../../macos/lib
    cp ./build/bin/*.metal ../../macos/lib
    cp ./build/bin/ggml-common.h ../../macos
    cp ./build/bin/ggml-metal-impl.h ../../macos/lib
  CMD

  s.vendored_libraries = 'lib/libggml-base.dylib', 'lib/libggml-blas.dylib', 'lib/libggml-cpu.dylib', 'lib/libggml-metal.dylib', 'lib/libggml.dylib', 'lib/libllama.dylib', 'lib/libllava_shared.dylib'
  s.frameworks = 'Foundation', 'Metal', 'MetalKit'
  s.platform = :osx, '10.15'
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
  s.script_phases = [
    {
      :name => 'Build Metal Library',
      :input_files => ["${PODS_TARGET_SRCROOT}/lib/ggml-metal.metal"],
      :output_files => ["${METAL_LIBRARY_OUTPUT_DIR}/default.metallib"],
      :execution_position => :after_compile,
      :script => <<-SCRIPT
      set -e
      set -u
      set -o pipefail
      xcrun metal -target "air64-${LLVM_TARGET_TRIPLE_VENDOR}-${LLVM_TARGET_TRIPLE_OS_VERSION}${LLVM_TARGET_TRIPLE_SUFFIX:-\"\"}" -ffast-math -std=ios-metal2.3 -o "${METAL_LIBRARY_OUTPUT_DIR}/default.metallib" ${PODS_TARGET_SRCROOT}/lib/ggml-metal.metal
      SCRIPT
    }
  ]
end