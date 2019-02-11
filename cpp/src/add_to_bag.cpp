#include <rosbag/bag.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
#include <stdio.h>

int main(int argc, char* argv[])
{

    std::string bag_fl;
    if (argc < 2)
        bag_fl = "test.bag";
    else
        bag_fl = argv[1];

    ros::Time::init();
    rosbag::Bag bag;
    bag.open(bag_fl, rosbag::bagmode::Write);
    std_msgs::String str;
    str.data = std::string("foo");
    std_msgs::Int32 i;
    i.data = 42;
    bag.write("chatter", ros::Time::now(), str);
    bag.write("numbers", ros::Time::now(), i);
    bag.close();
}
