#include "unet.h"
#include <time.h>

#include <opencv2/highgui.hpp>


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printf("Must provide image file!\n");
        return -1;
    }
    
    cv::Mat im = cv::imread(argv[1]);
    if (!im.data)
    {
        printf("Could not open image file: %s\n", argv[1]);
        return -1;
    }


    clock_t t0 = clock();
    UNet unet;
    printf("Initialized in %f ms\n", 1000 * (double)(clock()-t0) / CLOCKS_PER_SEC);

    cv::Mat mask;

    t0 = clock();
    unet.run(im, mask);
    printf("Inference took %f ms\n", 1000 * (double)(clock()-t0) / CLOCKS_PER_SEC);
    
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE );

    cv::Mat im2;
    cv::resize(im, im, cv::Size(160, 120));
    cv::cvtColor(im, im, cv::COLOR_BGR2GRAY);
    cv::hconcat(im, mask, im2);
    cv::imshow("Display window", im2);              

    cv::waitKey(0);                                          

    return 0;
}





















