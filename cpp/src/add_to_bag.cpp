#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "unet.h"
#include "tracker.h"

int main(int argc, char* argv[])
{

    // Load the bag that we want to process
    std::string bag_fl;
    if (argc < 2)
        bag_fl = "test.bag";
    else
        bag_fl = argv[1];

    // IT WE SHOULD APPEND TO THE BAG FILE
    // TODO: WE SHOULD PASS THIS FROM THE COMMAND LINE
    bool do_append2bag = false;

    // Topics that we want to compute masks for
    std::vector<std::string> topics;
    topics.push_back("/cam0/image_raw");
    topics.push_back("/cam1/image_raw");

    // Debug printing
    std::cout << "Loading ROS Bag File.." << std::endl;
    for(size_t i=0; i<topics.size(); i++) {
        std::cout << "\ttopic_" << i << " = " << topics.at(i) << std::endl;
    }

    // Open the bag, select the image topics
    rosbag::Bag bag;
    bag.open(bag_fl, rosbag::bagmode::Read);
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    // Loop through and get the list of image messages
    std::vector<std::vector<sensor_msgs::ImageConstPtr>> cams_msgs;
    cams_msgs.resize(topics.size());
    for (rosbag::MessageInstance m : view) {
        // Get the message and the topic it is on
        sensor_msgs::ImageConstPtr msg = m.instantiate<sensor_msgs::Image>();
        std::string topic_msg = m.getTopic();
        // check if it matches one of our topics
        for(size_t i=0; i<topics.size(); i++) {
            if (topic_msg == topics.at(i)) {
                cams_msgs.at(i).push_back(msg);
                break;
            }
        }
    }
    bag.close();

    // Debug printing
    std::cout << "Done Loading Bag, Now Process Masks!!" << std::endl;
    for(size_t i=0; i<topics.size(); i++) {
        std::cout << "\tnum messages " << i << " = " << cams_msgs.at(i).size() << std::endl;
    }



    // Create the UNET object!!
    // TODO: pass the location of the network here...
    UNet unet;

    // Re-open the bag
    bag.open(bag_fl, rosbag::bagmode::Read);

    // Nice debug display of the results
    cv::Mat im, mask;
    std::string windowname = "Network Input | Network Mask Output";
    cv::namedWindow(windowname, cv::WINDOW_AUTOSIZE);

    // Process the left and right images
    for(size_t i=0; i<topics.size(); i++) {


        // Init our Kalman filter!
        initialize_kf();


        // Get this camera's messages
        std::vector<sensor_msgs::ImageConstPtr> cam_msgs = cams_msgs.at(i);
        std::cout << std::endl << "Working on Topic " << i << " ....." << std::endl;

        // Stats for average inference times
        double sum_inf = 0;
        double max_inf = 0;

        // Loop through this topic's messages
        for (size_t j=0; j<cam_msgs.size(); j++) {

            // Convert the original image into a opencv matrix
            try {
                cv_bridge::CvImageConstPtr image = cv_bridge::toCvCopy(cam_msgs.at(j), sensor_msgs::image_encodings::BGR8);
                im = image->image;
            } catch (cv_bridge::Exception& e) {
                fprintf(stderr, "cv_bridge exception: %s", e.what());
                bag.close();
                return -1;
            }

            // Run the actual network
            clock_t t0 = clock();
            cv::Rect bbox = unet.run(im, mask);

            // Get inference time and update stats
            double inference_time =  1000*((double)(clock()-t0) / CLOCKS_PER_SEC);
            sum_inf += inference_time;
            max_inf = std::max(max_inf,inference_time);

            // Print out debug messages
            std::cout << "Processing message " << j << " / " << cam_msgs.size()
                        << " (" << std::setprecision(4) << ((double)j/cam_msgs.size()*100) << "%)"
                        << " => inference took " << std::setprecision(4)  << inference_time << " ms"
                        << " | " << sum_inf/(j+1) << " ms avg | " << max_inf << " ms max" << std::endl;


            // Now write the mask image back to the bag if we need to
            // TODO: fix me...
            if(do_append2bag) {
                cv_bridge::CvImage out_msg;
                out_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC1;
                out_msg.header = cam_msgs.at(j)->header; 
                out_msg.image = 255*mask;
                /*
                bag.write(i==0?"/cam0/image_mask":"/cam1/image_mask",
                        ros::Time(msg->header.stamp.sec, msg->header.stamp.nsec),
                        out_msg);
                */
            }

            // Do our kalman filtering
            cv::Rect bbox_kf;
            bool success = update_kf(cam_msgs.at(j)->header.stamp.toSec(), bbox, bbox_kf);

            // Debug display of the  
            cv::Mat im2, mask2, imout;
            cv::resize(im, im2, cv::Size(320, 240));
            cv::resize(im, im2, cv::Size(320, 240));

            // copy the image that the box bounds
            cv::Mat im3(cv::Size(320, 240), im.type(), cv::Scalar(0));
            if(0 <= bbox_kf.x && 0 <= bbox_kf.width && bbox_kf.x + bbox_kf.width <= im3.cols && 0 <= bbox_kf.y && 0 <= bbox_kf.height && bbox_kf.y + bbox_kf.height <= im3.rows) {
                // select subimage    
                cv::Mat subimg = im2(bbox_kf).clone();
                // extract fast for debuging
                std::vector<cv::KeyPoint> corners;
                cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(15,true);
                detector->detect(subimg,corners,cv::Mat());
                auto it = corners.begin();
                while (it != corners.end()) {
                    cv::circle(subimg,(*it).pt,1,cv::Scalar(100,100,255),1);
                    ++it;
                }                               
                // copy the the larger image
                subimg.copyTo(im3(cv::Rect(160-bbox_kf.width/2, 120-bbox_kf.height/2, subimg.cols, subimg.rows)));
                cv::putText(im3, std::to_string(bbox_kf.width)+" by "+std::to_string(bbox_kf.height), cv::Point(10,20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar::all(255), 1, 1);
            }

            // draw bounding boxes
            cv::rectangle(im2, bbox, cv::Scalar(255,0,0), 1);
            if(success) cv::rectangle(im2, bbox_kf, cv::Scalar(0,255,0), 1);
            else cv::rectangle(im2, bbox_kf, cv::Scalar(0,0,255), 2);

            // finally fuse all the images together
            cv::cvtColor(255 * mask, mask2, cv::COLOR_GRAY2BGR);
            cv::hconcat(im2, mask2, imout);
            cv::hconcat(imout, im3, imout);
            cv::resize(imout, imout, cv::Size(6*320, 2*240));
            cv::imshow(windowname, imout);
            cv::waitKey(10);

        }
        std::cout << std::endl;
    }
    bag.close();
}
