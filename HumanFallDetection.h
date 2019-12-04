#include <opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<vector>
#include<cmath>
#include<iostream>
class HumanFallDetection
 {	
 private:
 cv::Mat segment;
 cv::Mat image;
 cv::Mat im_out;
 int x,y,w,h;
 cv::Mat eigenvalue;
 cv::Mat eigenvector;
 cv::Mat covariance_matrix;
 float theta;
 cv::Point centroid;
 cv::Mat kernel;
 int m_kernel;
 int f_kernel;
 std::string dir_path;
 public:
         HumanFallDetection(int ,int,std::string);
	 ~HumanFallDetection();
	void run();
	void segmentation(cv::Mat &,cv:: Mat &); 
	void morphology_operation(int , cv::Mat &);
	void convolve_generic(const cv::Mat &, cv::Mat &, const cv::Mat &);
	void bounding_box(cv::Mat &,int & ,int & ,int & ,int  & );
	void covariance_matrix_calculation(cv::Mat &,cv::Mat &,cv::Mat &,cv::Mat &,cv::Point & );
	void theta_calculation(const cv::Mat &,const cv::Mat &,const cv::Mat &,float &,cv::Point );
	void visualisation(float ,int ,int ,int ,int );
};

