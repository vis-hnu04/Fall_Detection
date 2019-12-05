#include<iostream>
#include "HumanFallDetection.h"
using namespace cv;
using namespace std;
void help(char** argv)
{
	cout << "\nThis program gets you started reading a sequence of images using cv::VideoCapture.\n"
		<< "Image sequences are a common way to distribute video data sets for computer vision.\n"
		<< "Usage: " << argv[0] << " <path to the first image in the sequence>\n"
		<< "example: " << argv[0] << " right%%02d.jpg\n"
		<< "q,Q,esc --quit\n"
		<< "\tThis is a starter sample\n"
		<< endl;
}

int main(int argc, char** argv)

{

if (argc != 2)
	{
		help(argv);
		return 1;
	}

	string arg = argv[1];

HumanFallDetection *test=new HumanFallDetecion(7,3,arg);
test->run();

}


