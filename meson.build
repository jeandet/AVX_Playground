project('AVX_Playground', 'cpp',
        default_options : ['cpp_std=c++20', 'buildtype=release'],
        license: 'GPL3',
        version: '0.1.0')

avx_args = [
    '-mavx',
    '-mavx2',
    '-mfma',
    '-mfma4',
    '-mavx512f',
    '-mavx512cd',
    '-mavx512dq',
    '-mavx512bw',
    '-mavx512vl',
    '-mavx512ifma',
    '-mavx512vbmi',
    '-mavx512vnni'
]

no_avx_args = [
    '-mno-avx',
    '-mno-avx2',
    '-mno-fma',
    '-mno-fma4',
    '-mno-sse',
    '-mno-avx512f',
    '-mno-avx512cd',
    '-mno-avx512dq',
    '-mno-avx512bw',
    '-mno-avx512vl',
    '-mno-avx512ifma',
    '-mno-avx512vbmi',
    '-mno-avx512vnni',
]

googleBenchmark = dependency('benchmark')

xsimd = dependency('xsimd')
thread_dep = dependency('threads')

comparison = executable('comparison', 'src/comparison.cpp','src/loops.cpp',
                        dependencies: [googleBenchmark, xsimd, thread_dep],
                        c_args: avx_args,
                        cpp_args: avx_args)
