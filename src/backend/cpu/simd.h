#ifdef __aarch64__
struct SimdTypes {
    template<int dim> using FIn = const madevent::TensorView<double, dim>;
    template<int dim> using IIn = const madevent::TensorView<long long, dim>;
    template<int dim> using BIn = const madevent::TensorView<bool, dim>;
    template<int dim> using FOut = madevent::TensorView<double, dim>;
    template<int dim> using IOut = madevent::TensorView<long long, dim>;
    template<int dim> using BOut = madevent::TensorView<bool, dim>;

}
#endif
