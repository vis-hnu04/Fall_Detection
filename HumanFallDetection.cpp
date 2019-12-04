#include "HumanFallDetection.h"
using namespace std;
using namespace cv;
HumanFallDetection::HumanFallDetection(int morphology_kernel=7,int filter_kernel_size=3,string path=NULL){
			m_kernel=morphology_kernel;
			f_kernel=filter_kernel_size;

			dir_path=path;
			cv::Mat kernel(f_kernel,f_kernel,CV_8UC1);
			}
HumanFallDetection::~HumanFallDetection(){};
void HumanFallDetection::run(){
                        cv::VideoCapture sequence(dir_path);
			sequence.read(image);
			resize(image, image, Size(image.cols / 3, image.rows / 3));
			segmentation(image,segment);
			Mat im_out=Mat::zeros(segment.rows, segment.cols,CV_8UC1);
			convolve_generic(segment,im_out,kernel);
			morphology_operation(m_kernel , im_out);
			bounding_box(im_out,x,y,w,h);
			covariance_matrix_calculation(im_out,eigenvalue,eigenvector,covariance_matrix,centroid);
			theta_calculation(im_out,eigenvalue,eigenvector,theta,centroid );
			visualisation(theta,x,y,w,h);
}
void HumanFallDetection::segmentation(cv::Mat &image,cv:: Mat &segment){
			Ptr<BackgroundSubtractor> pMOG2;
			pMOG2=createBackgroundSubtractorMOG2();
			pMOG2->apply(image,segment);}
void HumanFallDetection::convolve_generic(const cv::Mat &input,cv::Mat &output, const cv::Mat &kernel)
{
			if (input.empty() || kernel.empty())
				{
				std::cout << "One ore more inputs are empty!" << std::endl;
				return;
				}

			int rows = input.rows;
			int cols = input.cols;

	// create a float image initialized with zeros

			int kRows = kernel.rows;
			int kCols = kernel.cols;


	// calculate the normalisation factor from the filter kernel

	// perform a generic convolution with cropped edges
			kRows = kernel.rows;
			kCols = kernel.cols;

			int kHotspotX = kCols / 2;
			int kHotspotY = kRows / 2;

	// perform convolution
			for (int r = 0; r < (rows - kRows + 1); ++r)
				{
					uchar *pOutput = output.ptr<uchar>(r + kHotspotY) + kHotspotX;

					for (int c = 0; c < (cols - kCols + 1); ++c)
						{
			//float result = 0.0f;
							int count = 0;
							for (int kr = 0; kr < kRows; ++kr)
							{
								const uchar *pInput = input.ptr<uchar>(r + kr) + c;
								const signed char *pKernel = kernel.ptr<signed char>(kr);

								for (int kc = 0; kc < kCols; ++kc)
									{
									if (*pInput == 255) {
										count++;
										}
					//++pKernel;
									++pInput;
								}
							}

						if (count >5) {

							*pOutput = 255;
							}
					++pOutput;
					}
	}
}

void HumanFallDetection::morphology_operation(int kernel_size, Mat &im_out)
{
			Mat structuring_element = getStructuringElement(MORPH_ELLIPSE, Size(kernel_size, kernel_size));
			morphologyEx(im_out, im_out, MORPH_CLOSE, structuring_element);
}
void HumanFallDetection::bounding_box(Mat &im_out,int &x ,int &y ,int &w ,int &h ){
			Mat labels;
			Mat stats;
			Mat centroids;
			cv::connectedComponentsWithStats(im_out, labels, stats, centroids);
			int max_Area = 0;
			int maximum_Area_index;
			
			for (int i = 1; i < stats.rows; i++)
			{
				 x = stats.at<int>(Point(0, i));
				 y = stats.at<int>(Point(1, i));	
				 w = stats.at<int>(Point(2, i));
				 h = stats.at<int>(Point(3, i));
				int area = stats.at<int>(Point(4, i));
				if (area > max_Area) {
					max_Area = area;
					maximum_Area_index = i;
				}
				

			}
			 x = stats.at<int>(Point(0, maximum_Area_index));
			 y = stats.at<int>(Point(1, maximum_Area_index));
			 w = stats.at<int>(Point(2, maximum_Area_index));
			 h = stats.at<int>(Point(3, maximum_Area_index));
}
void HumanFallDetection::covariance_matrix_calculation(cv::Mat &im_out,cv::Mat &eigenvalue,cv::Mat &eigenvector,cv::Mat &covariance_matrix,cv::Point &centroid )
{
			Moments m = moments(im_out, true);
			float data[4]={m.mu20,m.mu11,m.mu11,m.mu02};
			covariance_matrix =cv::Mat(2,2,CV_32F,data);
			eigenvalue =cv::Mat(1, 2, CV_32F);
			eigenvector =cv::Mat(2, 2, CV_32F);
			eigen(covariance_matrix, eigenvalue, eigenvector);
                              //point p givs silhouette centroid
			centroid =cv::Point(m.m10 / m.m00, m.m01 / m.m00);	
}
void HumanFallDetection::theta_calculation(const cv::Mat &im_out,const cv::Mat &eigenvalue,const cv::Mat &eigenvector,float &theta,cv::Point centroid )
{
			float x0 = im_out.cols / 2;
			float y0 = im_out.rows / 2;
			float den = sqrt(pow((x0 - centroid.x), 2) + pow((y0 - centroid.y), 2));
			float vect[2] = { (x0 - centroid.x) / den, (y0 - centroid.y) / den };
        			
                        Mat vector2(1, 2, CV_32F, vect);

                            //finding ratio of eigen values
                        const float* eig_ptr1 = eigenvalue.ptr<float>(0);
			float eigenvalue11 = abs(eig_ptr1[0]);
			float eigenvalue22 = abs(eig_ptr1[1]);
			float lambda = (eigenvalue11 > eigenvalue22) ? (eigenvalue11 / eigenvalue22) : (eigenvalue22/ eigenvalue11);
			int   index = eigenvalue11 > eigenvalue22 ? 0 : 1;
			const float* eigvec_ptr1 = eigenvector.ptr<float>(index);
			float norl = sqrt((eigvec_ptr1[0] * eigvec_ptr1[0]) + (eigvec_ptr1[1] * eigvec_ptr1[1]));
			float ppp1[2] = { eigvec_ptr1[0] / norl,eigvec_ptr1[1] / norl };
			Mat pp1(1, 2, CV_32F, ppp1);
			theta = (vector2.dot(pp1));
}
void HumanFallDetection::visualisation(float theta ,int x,int y ,int w,int h ){
        		int 	fallframe=0;
			if (abs(theta)<0.8 )
                         {    
        			fallframe+=1;
                                if (fallframe>=20 && x!=0){
     					Rect rect(x, y, w, h);
			                Scalar color(0,0, 255);
			                rectangle(image, rect, color);
                                        putText(image,"fall detected",Point(x,y),cv::FONT_HERSHEY_DUPLEX,1.0,CV_RGB(200,200,150),2);
                                        //fallframe=0;}
                                                      }
                                else if (x==0){}
                                else
                                  {   Rect rect(x, y, w, h);
			              Scalar color(0,255, 0);
			              rectangle(image, rect, color);
                                   }  
                            }          
                               
                         else
                          { Rect rect(x, y, w, h);
			    Scalar color(0,255,0);	
			    rectangle(image, rect, color);
                             
                          fallframe=0;
			  }		
}
