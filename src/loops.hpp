#include "xsimd/xsimd.hpp"
#include <cstddef>
#include <vector>

namespace xs = xsimd;

auto fill_vector(auto&& vec, double start_value=0, double increment=1)
{
    std::decay_t<decltype(vec[0])> _start_value = start_value;
    std::decay_t<decltype(vec[0])> _increment = increment;
    for (auto& v : vec)
    {
        v = _start_value;
        _start_value += _increment;
    }
    return vec;
}


template <typename value_type>
struct xsimd_mean
{
    using vector_type = std::vector<value_type, xsimd::aligned_allocator<value_type>>;

    inline auto setup(std::size_t size) const noexcept
    {
        return std::tuple { fill_vector(vector_type(size)), fill_vector(vector_type(size)), vector_type(size, 0.) };
    }

    void operator()(const vector_type& a, const vector_type& b, vector_type& res) const noexcept
    {
        std::size_t size = a.size();
        constexpr std::size_t simd_size = xsimd::simd_type<value_type>::size;
        const std::size_t vec_size = size - size % simd_size;
        constexpr std::size_t unroll_count = 8;
        const std::size_t unrolled_size = vec_size - vec_size % (simd_size * unroll_count);

        for(std::size_t i = 0; i< unrolled_size; i += simd_size * unroll_count)
        {
            auto ba0 = xs::load_aligned(&a[i]);
            auto bb0 = xs::load_aligned(&b[i]);
            auto bres0 = (ba0 + bb0) / 2.;
            bres0.store_aligned(&res[i]);

            auto ba1 = xs::load_aligned(&a[i + simd_size]);
            auto bb1 = xs::load_aligned(&b[i + simd_size]);
            auto bres1 = (ba1 + bb1) / 2.;
            bres1.store_aligned(&res[i + simd_size]);

            auto ba2 = xs::load_aligned(&a[i + 2 * simd_size]);
            auto bb2 = xs::load_aligned(&b[i + 2 * simd_size]);
            auto bres2 = (ba2 + bb2) / 2.;
            bres2.store_aligned(&res[i + 2 * simd_size]);

            auto ba3 = xs::load_aligned(&a[i + 3 * simd_size]);
            auto bb3 = xs::load_aligned(&b[i + 3 * simd_size]);
            auto bres3 = (ba3 + bb3) / 2.;
            bres3.store_aligned(&res[i + 3 * simd_size]);

            auto ba4 = xs::load_aligned(&a[i + 4 * simd_size]);
            auto bb4 = xs::load_aligned(&b[i + 4 * simd_size]);
            auto bres4 = (ba4 + bb4) / 2.;
            bres4.store_aligned(&res[i + 4 * simd_size]);

            auto ba5 = xs::load_aligned(&a[i + 5 * simd_size]);
            auto bb5 = xs::load_aligned(&b[i + 5 * simd_size]);
            auto bres5 = (ba5 + bb5) / 2.;
            bres5.store_aligned(&res[i + 5 * simd_size]);

            auto ba6 = xs::load_aligned(&a[i + 6 * simd_size]);
            auto bb6 = xs::load_aligned(&b[i + 6 * simd_size]);
            auto bres6 = (ba6 + bb6) / 2.;
            bres6.store_aligned(&res[i + 6 * simd_size]);

            auto ba7 = xs::load_aligned(&a[i + 7 * simd_size]);
            auto bb7 = xs::load_aligned(&b[i + 7 * simd_size]);
            auto bres7 = (ba7 + bb7) / 2.;
            bres7.store_aligned(&res[i + 7 * simd_size]);

        }


        for (std::size_t i = unrolled_size; i < vec_size; i += simd_size)
        {
            auto ba = xs::load_aligned(&a[i]);
            auto bb = xs::load_aligned(&b[i]);
            auto bres = (ba + bb) / 2.;
            bres.store_aligned(&res[i]);
        }

        for (std::size_t i = vec_size; i < size; ++i)
        {
            res[i] = (a[i] + b[i]) / 2.;
        }
    }
};

template <typename value_type>
struct no_vect_mean
{

    using vector_type = std::vector<value_type>;

    inline auto setup(std::size_t size) const noexcept
    {
        return std::tuple { fill_vector(vector_type(size)), fill_vector(vector_type(size)), vector_type(size, 0.) };
    }

    void __attribute__((target("no-avx"), target("no-fma"), target("no-avx2"), target("no-sse2"))) operator()(
        const vector_type & a, const vector_type & b, vector_type & res) const noexcept
    {
        for (std::size_t i = 0; i < a.size(); i++)
        {
            res[i] = (a[i] + b[i]) / 2.;
        }
    }
};
