# TF Unet

This is a UNet implementation designed to detect cars in the cityscapes dataset. The upconv layers have been replaced with subpixel convolution, which is very fast. If you freeze a model and run it in the cpp directory, you can see that it runs in less than 20ms on average.
