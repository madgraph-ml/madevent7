
template<typename T>
KERNELSPEC void kernel_sample_discrete(
    FIn<T,0> r, IIn<T,0> option_count, IOut<T,0> output, FOut<T,0> det
) {

}

template<typename T>
KERNELSPEC void kernel_sample_discrete_probs(
    FIn<T,0> r, FIn<T,1> probs, IOut<T,0> output, FOut<T,0> det
) {

}

template<typename T>
KERNELSPEC void kernel_gather(
    IIn<T,0> index, FIn<T,1> choices, FOut<T,0> output
) {

}

template<typename T>
KERNELSPEC void kernel_gather_int(
    IIn<T,0> index, IIn<T,1> choices, IOut<T,0> output
) {

}

template<typename T>
KERNELSPEC void kernel_one_hot(
    IIn<T,0> index, IIn<T,0> option_count, FOut<T,1> output
) {

}
