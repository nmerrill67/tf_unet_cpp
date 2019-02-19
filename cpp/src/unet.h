#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/c/c_api.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/highgui/highgui.hpp"

TF_Buffer* read_tf_buffer(const char* file);                                          
void dealloc(void* data, size_t len, void* arg);
void free_buffer(void* data, size_t len);

class UNet
{
public:
    UNet();
    ~UNet();
    cv::Rect run(const cv::Mat& im, cv::Mat& out);

    TF_Graph* graph;
    TF_Status* status;
    TF_Session* sess;
    TF_Tensor* input;
    TF_Tensor* output;
    TF_Output in_op;
    TF_Output out_op;
    int w, h, c;
    int64_t dims[4];
};
