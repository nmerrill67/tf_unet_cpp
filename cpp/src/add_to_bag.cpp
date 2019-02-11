#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include <iostream>

#include <opencv2/highgui.hpp>

#include "unet.h"

int main(int argc, char* argv[])
{

    std::string bag_fl;
    if (argc < 2)
        bag_fl = "test.bag";
    else
        bag_fl = argv[1];

    rosbag::Bag bag;
    bag.open(bag_fl, rosbag::bagmode::Read);
    std::vector<std::string> topics;
    topics.push_back(std::string("/cam0/image_raw"));
    topics.push_back(std::string("/cam1/image_raw"));
    rosbag::View view(bag, rosbag::TopicQuery(topics));
    cv::Mat im, mask;
    std::vector<sensor_msgs::ImageConstPtr> cam0_msgs, cam1_msgs;

    for (rosbag::MessageInstance m : view)
    {   
        sensor_msgs::ImageConstPtr msg = m.instantiate<sensor_msgs::Image>();
        std::string t = m.getTopic();
        if (!t.compare("/cam0/image_raw"))
            cam0_msgs.push_back(msg);
        else
            cam1_msgs.push_back(msg);
    }
    
    printf("Read rosbag. Now appending masks...\n");

    bag.close();

    bag.open(bag_fl, rosbag::bagmode::Append);
    UNet unet;
    cv_bridge::CvImage out_msg;
    out_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC1; 
    int j = 0;
    for (int i = 0; i < 2; i++)
    {   
        printf("Working on cam%d\n", i);
        for (sensor_msgs::ImageConstPtr msg : i==0?cam0_msgs:cam1_msgs)
        {
            try {
                cv_bridge::CvImageConstPtr image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
                im = image->image;
            } catch (cv_bridge::Exception& e) {
                fprintf(stderr, "cv_bridge exception: %s", e.what());
                bag.close();
                return -1;
            }
            unet.run(im, mask);
            out_msg.header = msg->header; 
            out_msg.image = mask;
            bag.write(i==0?"/cam0/image_mask":"/cam1/image_mask",
                    ros::Time(msg->header.stamp.sec, msg->header.stamp.nsec),
                    out_msg);
        }
    }
    bag.close();
}
