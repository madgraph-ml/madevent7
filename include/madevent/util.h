#pragma once

#include <ranges>
#include <tuple>

namespace madevent {

template<class... Ts> struct Overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;

#if __cplusplus != 202302L
// Unfortunately nvcc does not support C++23 yet, so we implement our own zip function
// here (based on https://github.com/alemuntoni/zip-views), otherwise use the standard
// library function

namespace detail {

template <typename... Args, std::size_t... Index>
bool any_match_impl(
    const std::tuple<Args...>& lhs,
    const std::tuple<Args...>& rhs,
    std::index_sequence<Index...>
) {
    auto result = false;
    result = (... || (std::get<Index>(lhs) == std::get<Index>(rhs)));
    return result;
}

template <typename ... Args>
bool any_match(const std::tuple<Args...>& lhs, const std::tuple<Args...>& rhs) {
    return any_match_impl(lhs, rhs, std::index_sequence_for<Args...>{});
}

template <std::ranges::viewable_range... Rng>
class zip_iterator {
public:
    using value_type = std::tuple<std::ranges::range_reference_t<Rng>...>;

    zip_iterator() = delete;
    zip_iterator(std::ranges::iterator_t<Rng>&&... iters)
        : _iters{std::forward<std::ranges::iterator_t<Rng>>(iters)...}
    {}

    zip_iterator& operator++() {
        std::apply([](auto && ... args){ ((++args), ...); }, _iters);
        return *this;
    }

    zip_iterator operator++(int) {
        auto tmp = *this;
        ++*this;
        return tmp;
    }

    bool operator!=(const zip_iterator& other) const {
        return !(*this == other);
    }

    bool operator==(const zip_iterator& other) const {
        return any_match(_iters, other._iters);
    }

    value_type operator*() {
        return std::apply([](auto && ... args) {
            return value_type(*args...);
        }, _iters);
    }

private:
    std::tuple<std::ranges::iterator_t<Rng>...> _iters;
};

template <std::ranges::viewable_range... T>
class zipper
{
public:
    using zip_type = zip_iterator<T...>;

    template <typename... Args>
    zipper(Args&&... args) : _args{std::forward<Args>(args)...} {}

    zip_type begin() {
        return std::apply([](auto && ... args){
            return zip_type(std::ranges::begin(args)...);
            }, _args);
    }
    zip_type end() {
        return std::apply([](auto && ... args){
            return zip_type(std::ranges::end(args)...);
        }, _args);
    }

private:
    std::tuple<T ...> _args;
};

}

template <std::ranges::viewable_range... T>
auto zip(T&& ... t) {
    return detail::zipper<T...>{std::forward<T>(t)...};
}

#else //__cplusplus != 202302L

template <std::ranges::viewable_range... T>
auto zip(T&& ... t) {
    return std::views::zip(std::forward<T>(t)...);
}

#endif //__cplusplus != 202302L

}
