class Accessor {
public:
    Accessor(double* _pointer, int _stride) : pointer(_pointer), stride(_stride) {}
    double& operator*() const { return *pointer; }
    const double& operator[](size_t index) const { return pointer[index * stride]; }
    double& operator[](size_t index) const { return pointer[index * stride]; }

private:
    double* pointer;
    int stride;
};
