#include "loops.hpp"
#include <benchmark/benchmark.h>

template <typename mean_function_t>
static void BM_mean(benchmark::State& state)
{
    // Perform setup here
    mean_function_t mean_function;
    auto [a, b, res] = mean_function.setup(state.range(0));
    for (auto _ : state)
    {
        mean_function(a, b, res);
    }
    benchmark::DoNotOptimize(res);
    state.SetComplexityN(state.range(0));
    state.SetItemsProcessed(state.iterations() * state.range(0));
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(decltype(a[0])) * 3);
    state.counters["OPS"]
        = benchmark::Counter(state.iterations() * state.range(0), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_mean<no_vect_mean<double>>)->RangeMultiplier(8)->Range(256, 1024 * 1024 * 128);
BENCHMARK(BM_mean<xsimd_mean<double>>)->RangeMultiplier(8)->Range(256, 1024 * 1024 * 128);

BENCHMARK(BM_mean<no_vect_mean<float>>)->RangeMultiplier(8)->Range(256, 1024 * 1024 * 128);
BENCHMARK(BM_mean<xsimd_mean<float>>)->RangeMultiplier(8)->Range(256, 1024 * 1024 * 128);

BENCHMARK(BM_mean<no_vect_mean<std::int16_t>>)->RangeMultiplier(8)->Range(256, 1024 * 1024 * 128);
BENCHMARK(BM_mean<xsimd_mean<std::int16_t>>)->RangeMultiplier(8)->Range(256, 1024 * 1024 * 128);

BENCHMARK_MAIN();
