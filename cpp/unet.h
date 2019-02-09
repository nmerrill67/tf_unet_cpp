#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/c/c_api.h>
#include "opencv2/core.hpp"
    
void dealloc(void* data, size_t len, void* arg);

class UNet
{
public:
    UNet();
    ~UNet();
    void run(const cv::Mat& im, cv::Mat& out);

    TF_Graph* graph;
    TF_Status* status;
    TF_Session* sess;
    TF_Tensor* input;
    TF_Tensor* output;
    TF_Output in_op;
    TF_Output out_op;
    int w, h, c;
};
