//OpenCV Headers

#include "opencv2/highgui/highgui_c.h"
#include <cv.h>
#include "opencv.hpp"  
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <cmath>
#include "math.h"
using namespace std;
using namespace cv;

struct seg
{
	int Partnum;
	vector<Point> segAB;
	vector<Point> contourA;
	vector<Point> contourB;
	vector<Point> scontourA;
	vector<Point> scontourB;
	vector<Point> tabA;
	vector<Point> tabB;

	vector<Point> center;
	vector<Point> contour;
	vector<double> curvatureA;
	vector<double> curvatureB;
	vector<double> curvatureC;

	Point head;
	Point tail;
	Point mid;
	Point topleft;
	Point botright;

	float averageLength;
	float Length;
	float width;
	float volume;

	Mat I;
};

class WormTrack
{
public:
	WormTrack(const string path, Mat src, Mat src1, int filename, float m_sca, int m_guass, int m_thre, int m_mor, int partnum, Point A, Point B, seg* m_seg, bool& m_ht, float segmentLength);
	~WormTrack();

	std::string getFilename() const;
public:
	Mat morphology_element;
	Mat Orignal_img;
	Mat Pre_img;
	Mat img;

	Point scale_topLeft;
	Point scal_BottomRighe ;

private:
	int guass;
	int thre;
	int mor;
	float scale;
	int partnum;
	bool ht;
	int image_num;
	string path;
	float averageLength;
	Point A, B; // to determine ROI
	Mat img_contour;
	vector<Vec4i> hierarchy;
	vector<Point> comp_contours;

	vector<Point> boundA;
	vector<Point> boundB;
	vector<Point> centerline;
	vector<Point> recenterline;
	vector<Point> lookuptab;

	Point pre_head, pre_tail;
	Point head, tail;
	Point2f mc;

public:
	int Resize();
	seg segment;
	seg output;

	
	int PreProcess();
	int PreProcessDF();
	int Contour();
	int Analysis();

	int Segment();

	float Lengthcalculation();
	float Widthcalculation();
	float Volumecalculation();
	int Curvaturecalculation();
	void FindHeadTail(vector<Point>* input, vector<Point>& leftside, vector<Point>& rightside, Point &head, Point &tail, const int longvector, const int shortvector, float PercentLength);
private:
	void RemoveSmallContours(vector<vector<Point>> *rough_contours, vector<Vec4i> hierarchy, vector<Point> &contours);
	void SmoothSequence(const vector<Point>* input, vector<Point>& output, double sigma);
	void CreateGaussianKernel(double sigma, int **kernel, int *klength, int *normfactor);
	void ConvolveInt1D(const vector<Point>* input, vector<Point>& output, int length, int *kernel, int klength, int normfactor);
	
	void ResampleByOmit(vector<Point>* input, vector<Point>& output, int pointnum);
	int ResampleDist(vector<Point>* input, vector<Point>& output, vector<Point>& lookuptab, int pointnum);
	int ResampleDist_2f(vector<Point>* input, vector<Point2f>& output, vector<Point>& lookuptab, int pointnum);
	int FindCenterline(vector<Point>* inputleft, vector<Point>* inputright, vector<Point>& centerline);
	float Dist(Point a, Point b);
	float SquareDist(Point a, Point b);
	void SegmentWorm(vector<Point>* ReCenterline, vector<Point>* Centerline, vector<Point>* lookuptab, vector<Point>* BoundA, vector<Point>* BoundB, vector<Point>* Contours, seg* segment);

	int FindPerpPoint(vector<Point>* input, Point x, Point targent, Point &result, int partnum, int startindex, int endindex);
	void CompensateOffset(vector<Point>* input, vector<Point>* output, Point A);

	void newFindCenterline(vector<Point>* BoundA, vector<Point>* BoundB, vector<Point>* centerline, int pointnum, int step);
	void newSegmentWorm(vector<Point>* Centerline, vector<Point>* BoundA, vector<Point>* BoundB, seg* segment);
	int curvature(vector<Point>* input, vector<double>& output, int step);
};
void CalculateAngleSum(const string &Path, int j, seg &segment, Point &prevHead, Point &prevPharynx, Point &m_A, Point &m_B, bool switchht, double &angleSumHead, int partnum);
void ProcessImage(int j, const string &Path, seg &segment, Point &prevHead, Point &prevPharynx, Point &m_A, Point &m_B, bool switchht, double &angleSumHead, int partnum);