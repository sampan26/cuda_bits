#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
    CHECK_CUDA(x);                                                                                                       \
    CHECK_CONTIGUOUS(x)

typedef void SoftmaxFn(const float *input, float *out, int M, int N);

SoftmaxFn softmax_v1;
SoftmaxFn softmax_v2;
SoftmaxFn softmax_v3;

template <SoftmaxFn softmax_fn> torch::Tensor softmax_pt(torch::Tensor A) {
    CHECK_INPUT(A);
    int M = A.size(0);
    int N = A.size(1);
    torch::Tensor B = torch::empty({M, N}, A.options());
    softmax_fn(A.data_ptr<float>(), B.data_ptr<float>(), M, N);
    return B;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_v1", &softmax_pt<softmax_v1>, "Softmax v1");
    m.def("softmax_v2", &softmax_pt<softmax_v2>, "Online softmax");
    m.def("softmax_v3", &softmax_pt<softmax_v3>, "Online softmax shared");
}