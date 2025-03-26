project('AVX_Playground', 'cpp',
        default_options : ['cpp_std=c++20', 'buildtype=release'],
        license: 'GPL3',
        version: '0.1.0')

avx_args = [
    '-mavx',
]

no_avx_args = [
    '-mno-avx'
]

googleBenchmark = dependency('benchmark', default_options: {'tests':'disabled'})

xsimd = dependency('xsimd')
thread_dep = dependency('threads')

comparison = executable('comparison', 'src/comparison.cpp','src/loops.cpp',
                        dependencies: [googleBenchmark, xsimd, thread_dep],
                        c_args: avx_args,
                        cpp_args: avx_args)
