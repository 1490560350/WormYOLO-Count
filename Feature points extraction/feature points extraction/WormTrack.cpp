#include "stdafx.h"
#include "WormTrack.h"
#include "math.h"
#include <set>
#include <iostream>
#include <opencv2/opencv.hpp>
void CalculateAngleSum(const string &Path, int j, seg &segment, Point &prevHead, Point &prevPharynx, Point &m_A, Point &m_B, bool switchht, double &angleSumHead, int partnum, float segmentLength);

void ProcessImage(int j, const string &Path, seg &segment, Point &prevHead, Point &prevPharynx, Point &m_A, Point &m_B, bool switchht, double &angleSumHead, int partnum, float segmentLength);
WormTrack::WormTrack(string Path, Mat src, Mat src1, int filename, float m_sca, int m_guass, int m_thre, int m_mor, int m_partnum, Point m_A, Point m_B, seg* m_seg, bool& m_ht, float segmentLength)
{
	path = Path;
	Orignal_img = src;
	image_num = filename;
	Pre_img = src1;
	scale = m_sca;
	guass = m_guass;
	thre = m_thre;
	mor = m_mor;
	partnum = m_partnum;
	averageLength = segmentLength;
	A = m_A;
	B = m_B;
	ht = m_ht;

	std::cout << "0A: " << A.x << ", y: " << A.y << std::endl;

	m_ht = false;


	pre_head = m_seg->head;
	pre_tail = m_seg->tail;
	morphology_element = getStructuringElement(1, Size(mor, mor), Point(-1, -1));



}

WormTrack::~WormTrack()
{
	morphology_element.~Mat();

	img.~Mat();
	img_contour.~Mat();
}

int WormTrack::Resize()
{
	Mat pre_img;
	scale_topLeft = A;
	pre_img = Orignal_img;

	A.x = A.x > 0 ? A.x : 0;
	A.y = A.y > 0 ? A.y : 0;
	B.x = B.x < 2 * pre_img.size().width ? B.x : pre_img.size().width;
	B.y = B.y < 2 * pre_img.size().height ? B.y : pre_img.size().height;

	Rect rect(A, B);

	Mat image = pre_img(rect);

	scale_topLeft = A;


	
	Point scal_BottomRighe = B;
		
		if (scale != 1)
		{
			Mat scale_img(image.size().height * scale, image.size().width * scale, CV_8UC1);
			resize(image, scale_img, scale_img.size());


			image = scale_img;
		}

	img = image.clone();

	pre_img.~Mat();
	image.~Mat();
	return 1;
}

int WormTrack::PreProcess()
{

	Resize();

	if (guass)
	{
		GaussianBlur(img, img, Size(guass, guass), 1, 1);
	}

	threshold(img, img, thre, 255, CV_THRESH_BINARY_INV);

	dilate(img, img, morphology_element);
	erode(img, img, morphology_element);
	//imshow(" Image", img);
	//waitKey(0);
	return 1;
}

int WormTrack::PreProcessDF()
{

	Resize();

	if (guass)
	{
		GaussianBlur(img, img, Size(guass, guass), 1, 1);
	}

	threshold(img, img, thre, 255, CV_THRESH_BINARY);

	dilate(img, img, morphology_element);
	erode(img, img, morphology_element);

	return 1;
}

int WormTrack::Contour()
{
	vector<vector<Point>> rough_contours;
	vector<Point> smooth_contours, contours, dist_contour, tab;

	findContours(img, rough_contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	if (rough_contours.size() == 0)
	{
		return 0;
	}

	RemoveSmallContours(&rough_contours, hierarchy, contours);

	SmoothSequence(&contours, smooth_contours, 3);

	vector<Point> pre_contour;
	pre_contour.assign(smooth_contours.begin() + 1, smooth_contours.end());
	pre_contour.push_back(smooth_contours[0]);


	int num = pre_contour.size() < 300 ? pre_contour.size() : 300;
	ResampleDist(&pre_contour, dist_contour, tab, num);
	CompensateOffset(&dist_contour, &comp_contours, A);

	Moments mu = moments(contours, true);
	mc = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);

	return 1;
}

int WormTrack::Analysis()
{
	if (comp_contours.size() == 0)
	{
		return 0;
	}

	FindHeadTail(&comp_contours, boundA, boundB, head, tail, comp_contours.size() / 40 > 1 ? comp_contours.size() / 40 : 1, comp_contours.size() / 100 > 1 ? comp_contours.size() / 100 : 1, 0.2);

	if (image_num != 40)
	{
		std::cout << "image_num: " << image_num << std::endl;


		for (auto &pt : boundA)
		{
			pt.x = (pt.x - scale_topLeft.x) / scale + scale_topLeft.x;
			pt.y = (pt.y - scale_topLeft.y) / scale + scale_topLeft.y;
		}
		for (auto &pt : boundB)
		{
			pt.x = (pt.x - scale_topLeft.x) / scale + scale_topLeft.x;
			pt.y = (pt.y - scale_topLeft.y) / scale + scale_topLeft.y;
		}

		for (auto &pt : comp_contours)
		{
			pt.x = (pt.x - scale_topLeft.x) / scale + scale_topLeft.x;
			pt.y = (pt.y - scale_topLeft.y) / scale + scale_topLeft.y;
		}


		head.x = (head.x - scale_topLeft.x) / scale + scale_topLeft.x;
		head.y = (head.y - scale_topLeft.y) / scale + scale_topLeft.y;
		tail.x = (tail.x - scale_topLeft.x) / scale + scale_topLeft.x;
		tail.y = (tail.y - scale_topLeft.y) / scale + scale_topLeft.y;
		std::cout << "scale_topLeft: " << scale_topLeft.x << ", y: " << scale_topLeft.y << std::endl;

		segment.mid = Point((mc.x - scale_topLeft.x) / scale + scale_topLeft.x, (mc.y - scale_topLeft.y) / scale + scale_topLeft.y);
		A.x = (A.x - scale_topLeft.x) / scale + scale_topLeft.x;
		A.y = (A.y - scale_topLeft.y) / scale + scale_topLeft.y;
		B.x = (B.x - scale_topLeft.x) / scale + scale_topLeft.x;
		B.y = (B.y - scale_topLeft.y) / scale + scale_topLeft.y;
		std::cout << "Scaled Head Coordinates - x: " << A.x << ", y: " << A.y << std::endl;
		std::cout << "Scaled Tail Coordinates - x: " << B.x << ", y: " << B.y << std::endl;
	}
	else
	{
		segment.mid = Point(mc.x, mc.y);
	}



	segment.contourA = boundA;
	segment.contourB = boundB;
	segment.contour = comp_contours;
	segment.Partnum = partnum;
	segment.topleft = A;
	segment.botright = B;
	segment.I = Orignal_img;
	segment.head = head;
	segment.tail = tail;

	ResampleDist(&boundA, segment.scontourA, segment.tabA, partnum);
	ResampleDist(&boundB, segment.scontourB, segment.tabB, partnum);

	newFindCenterline(&segment.contourA, &segment.contourB, &centerline, partnum, 3);

	segment.center = centerline;

	return 1;
}


float WormTrack::Lengthcalculation()
{
	if (comp_contours.size() == 0)
	{
		return 0;
	}

	float length = 0;

	for (int k = 1; k < centerline.size(); k++)
	{
		length += Dist(centerline[k - 1], centerline[k]);
	}

	segment.Length = length;
	return length;
}




float WormTrack::Widthcalculation()
{
	if (comp_contours.size() == 0)
	{
		return 0;
	}

	int mid_index = (int)(segment.Partnum / 2 + 0.5);
	Point targent = segment.center[mid_index + 3] - segment.center[mid_index - 3];
	Point resultA, resultB;

	FindPerpPoint(&segment.contourA, segment.center[mid_index], targent, resultA, partnum - 1, partnum * 0.4, partnum * 0.6);
	FindPerpPoint(&segment.contourB, segment.center[mid_index], targent, resultB, partnum - 1, partnum * 0.4, partnum * 0.6);

	float width = Dist(resultA, resultB);

	segment.width = width;
	return width;
}

float WormTrack::Volumecalculation()
{
	if (comp_contours.size() == 0)
	{
		return 0;
	}

	vector<Point> lookuptabA, lookuptabB;
	vector<Point2f> reboundA2f, reboundB2f;
	ResampleDist_2f(&boundA, reboundA2f, lookuptabA, (partnum < centerline.size()) ? partnum : centerline.size());
	ResampleDist_2f(&boundB, reboundB2f, lookuptabB, (partnum < centerline.size()) ? partnum : centerline.size());

	double volume = 0;
	float r1 = 0;
	float r2 = 0;
	float h1 = 0;
	float h2 = 0;
	float r = 0;
	float h = 0;

	for (int i = 0; i < partnum; i++)
	{
		r1 = Dist(reboundA2f[i], reboundB2f[i]);
		r2 = Dist(reboundA2f[i + 1], reboundB2f[i + 1]);
		r = 0.5*(r1 + r2);

		h1 = Dist(reboundA2f[i], reboundA2f[i + 1]);
		h2 = Dist(reboundB2f[i], reboundB2f[i + 1]);
		h = 0.5*(h1 + h2);

		volume += (3.1415 * r * r) * h;
	}

	segment.volume = volume;
	return volume;
}


int WormTrack::Curvaturecalculation()
{
	curvature(&segment.scontourA, segment.curvatureA, 12);
	curvature(&segment.scontourB, segment.curvatureB, 12);
	curvature(&segment.center, segment.curvatureC, 12);
	return 1;
}

int WormTrack::curvature(vector<Point>* input, vector<double>& output, int step)
{
	if (input->size() == 0)
	{
		return 0;
	}

	vector<Point> scenter = *input;

	vector< double > vecCurvature(scenter.size());

	auto frontToBack = scenter.front() - scenter.back();
	bool isClosed = ((int)max(abs(frontToBack.x), abs(frontToBack.y))) <= 1;

	cv::Point2f pplus, pminus;
	cv::Point2f f1stDerivative, f2ndDerivative;
	for (int i = 0; i < scenter.size(); i++)
	{
		const cv::Point2f& pos = scenter[i];

		int maxStep = step;
		if (!isClosed)
		{
			maxStep = min(min(step, i), (int)scenter.size() - 1 - i);
			if (maxStep == 0)
			{
				vecCurvature[i] = std::numeric_limits<double>::infinity();
				continue;
			}
		}

		int iminus = i - maxStep;
		int iplus = i + maxStep;
		pminus = scenter[iminus < 0 ? iminus + scenter.size() : iminus];
		pplus = scenter[iplus > scenter.size() ? iplus - scenter.size() : iplus];

		f1stDerivative.x = (pplus.x - pminus.x) / (iplus - iminus);
		f1stDerivative.y = (pplus.y - pminus.y) / (iplus - iminus);
		f2ndDerivative.x = (pplus.x - 2 * pos.x + pminus.x) / ((iplus - iminus) / 2 * (iplus - iminus) / 2);
		f2ndDerivative.y = (pplus.y - 2 * pos.y + pminus.y) / ((iplus - iminus) / 2 * (iplus - iminus) / 2);

		double curvature2D;
		double divisor = f1stDerivative.x*f1stDerivative.x + f1stDerivative.y*f1stDerivative.y;
		if (std::abs(divisor) > 10e-8)
		{
			curvature2D = (f2ndDerivative.y*f1stDerivative.x - f2ndDerivative.x*f1stDerivative.y) /
				pow(divisor, 3.0 / 2.0);
		}
		else
		{
			curvature2D = std::numeric_limits<double>::infinity();
		}

		vecCurvature[i] = curvature2D;

	}

	output = vecCurvature;

	return 1;
}

int WormTrack::ResampleDist_2f(vector<Point>* input, vector<Point2f>& output, vector<Point>& lookuptab, int pointnum)
{
	float *dist, totaldist = 0;
	dist = (float *)malloc(input->size() * sizeof(float));
	dist[0] = 0;

	for (int i = 1; i < input->size(); i++)
	{
		totaldist += Dist((*input)[i - 1], (*input)[i]);
		dist[i] = totaldist;
	}

	float averagelength = totaldist / pointnum;
	float length = 0;
	float weightA, weightB;
	Point2f pt;
	Point lookup;

	int preindex = 0, currindex = 1;

	output.push_back((*input)[0]);
	lookuptab.push_back(Point(0, 0));


	for (int i = 1; i < pointnum; i++)
	{
		length = i * averagelength < totaldist ? i * averagelength : totaldist;

		while (length < dist[preindex] || length > dist[currindex])
		{
			preindex++;
			currindex++;
		}

		weightA = (dist[currindex] - length) / (dist[currindex] - dist[preindex]);
		weightB = 1 - weightA;

		pt.x = (weightA * (*input)[preindex].x + weightB * (*input)[currindex].x + 0.5);
		pt.y = (weightA * (*input)[preindex].y + weightB * (*input)[currindex].y + 0.5);

		output.push_back(pt);


		lookup.x = i;
		lookup.y = weightA > weightB ? preindex : currindex;
		lookuptab.push_back(lookup);
	}

	output.push_back((*input)[input->size() - 1]);
	lookuptab.push_back(Point(pointnum, input->size() - 1));

	free(dist);
	return 1;
}

int WormTrack::Segment()
{

	if (comp_contours.size() == 0)
	{
		return 0;
	}


	newSegmentWorm(&centerline, &boundA, &boundB, &segment);

	return 1;
}


void WormTrack::RemoveSmallContours(vector<vector<Point>> *rough_contours, vector<Vec4i> hierarchy, vector<Point> &contours)
{
	if (!rough_contours->empty() && !hierarchy.empty())
	{
		int idx = 0;
		int maxregion = (*rough_contours)[idx].size();
		int regionnumber = idx;

		for (; idx >= 0; idx = hierarchy[idx][0])
		{
			if ((*rough_contours)[idx].size() > maxregion)
			{
				maxregion = (*rough_contours)[idx].size();
				regionnumber = idx;
			}
		}
		contours = (*rough_contours)[regionnumber];

	}

}

void WormTrack::SmoothSequence(const vector<Point>* input, vector<Point>& output, double sigma)
{
	int *kernel, klength, normfactor;
	CreateGaussianKernel(sigma, &kernel, &klength, &normfactor);
	ConvolveInt1D(input, output, input->size(), kernel, klength, normfactor);
}

void WormTrack::CreateGaussianKernel(double sigma, int **kernel, int *klength, int *normfactor)
{
	int ll, ul, x;
	double n;
	ll = (int)(-3 * sigma) - 1;
	ul = (int)(3 * sigma) + 1;
	*klength = ul - ll + 1;
	*kernel = (int*)malloc(*klength * sizeof(int));

	*normfactor = 0;
	n = exp(-1.0*ll*ll / (2 * sigma*sigma));
	for (x = 0; x < *klength; x++) {
		(*kernel)[x] = (int)(exp(-1.0*(x + ll)*(x + ll) / (2 * sigma*sigma)) / n + 0.5);
		*normfactor += (*kernel)[x];
	}
}

void WormTrack::ConvolveInt1D(const vector<Point>* input, vector<Point>& output, int length, int *kernel, int klength, int normfactor)
{
	int j, k, ind, anchor, sumx, sumy;
	Point *pt;
	pt = (Point*)malloc(length * sizeof(Point));
	anchor = klength / 2;
	for (j = 0; j < length; j++) {
		sumx = sumy = 0;
		for (k = 0; k < klength; k++) {
			ind = j + k - anchor;
			ind = ind > 0 ? ind : 0;
			ind = ind < length ? ind : (length - 1);
			sumx = sumx + (*input)[ind].x * kernel[k];
			sumy = sumy + (*input)[ind].y * kernel[k];
		}

		int x = (int)(1.0*sumx / normfactor + 0.5);
		int y = (int)(1.0*sumy / normfactor + 0.5);

		pt[j].x = (int)(1.0*sumx / normfactor + 0.5);
		pt[j].y = (int)(1.0*sumy / normfactor + 0.5);

	}
	output = vector<Point>(pt, pt + length);

	free(pt);
}



void WormTrack::FindHeadTail(vector<Point>* input, vector<Point>& leftside, vector<Point>& rightside, Point &head, Point &tail, const int longvector, const int shortvector, float PercentLength)
{
	Point *pt_fa, *pt_ba, *pt_fa_s, *pt_ba_s;
	int *dotproduct_1;
	int *dotproduct_2;
	int *dotproduct;

	int totallength = input->size();

	bool flag = true;
	pt_fa = (Point *)malloc(totallength * sizeof(Point));
	pt_ba = (Point *)malloc(totallength * sizeof(Point));
	dotproduct_1 = (int *)malloc(totallength * sizeof(int));
	dotproduct_2 = (int *)malloc(totallength * sizeof(int));
	dotproduct = (int *)malloc(totallength * sizeof(int));

	for (int i = 0; i < totallength; i++)
	{

		pt_fa[i] = (*input)[(i + longvector) % totallength] - (*input)[i];
		pt_ba[i] = (*input)[(i - longvector + totallength) % totallength] - (*input)[i];
		dotproduct_1[i] = pt_fa[i].x * pt_ba[i].x + pt_fa[i].y * pt_ba[i].y;

		pt_fa[i] = (*input)[(i + shortvector) % totallength] - (*input)[i];
		pt_ba[i] = (*input)[(i - shortvector + totallength) % totallength] - (*input)[i];
		dotproduct_2[i] = pt_fa[i].x * pt_ba[i].x + pt_fa[i].y * pt_ba[i].y;

		dotproduct[i] = 1.5 * dotproduct_1[i] + longvector / shortvector* dotproduct_2[i];
	}



	int MostCurvy = dotproduct[0];
	int indextail = 0;
	for (int i = 0; i < totallength; i++)
	{
		if (MostCurvy < dotproduct[i])
		{
			MostCurvy = dotproduct[i];
			indextail = i;
		}
	}

	tail = (*input)[indextail];


	int searchlength = totallength*PercentLength;
	int SecondMostCurvyIndex = (indextail + totallength / 2) % totallength;
	int StartIndex = (SecondMostCurvyIndex - searchlength / 2 + totallength) % totallength;
	int EndIndex = (SecondMostCurvyIndex + searchlength / 2 + totallength) % totallength;


	if (abs(StartIndex - EndIndex) < totallength - abs(StartIndex - EndIndex))
	{
		StartIndex = StartIndex < EndIndex ? StartIndex : EndIndex;
	}
	else
	{
		StartIndex = StartIndex < EndIndex ? EndIndex : StartIndex;
	}

	int k = 0;
	int indexhead = SecondMostCurvyIndex;
	int SecondMostCurvy = dotproduct[SecondMostCurvyIndex];
	for (int i = 0; i < searchlength; i++)
	{
		k = (StartIndex + i) % totallength;
		if (SecondMostCurvy < dotproduct[k])
		{
			SecondMostCurvy = dotproduct[k];
			indexhead = k;
		}
	}
	head = (*input)[indexhead];

	if (indexhead > indextail)
	{
		StartIndex = indextail;
		EndIndex = indexhead;
		vector <Point> temp, temp_reverse;

		leftside.assign(input->begin() + EndIndex, input->end());
		temp.assign(input->begin(), input->begin() + StartIndex + 1);
		leftside.insert(leftside.end(), temp.begin(), temp.end());


		temp_reverse.assign(input->begin() + StartIndex, input->begin() + EndIndex + 1);
		vector <Point> rightside_1(temp_reverse.rbegin(), temp_reverse.rend());
		rightside.assign(rightside_1.begin(), rightside_1.end());
	}
	else
	{
		StartIndex = indexhead;
		EndIndex = indextail;
		vector <Point> temp, temp_reverse;

		leftside.assign(input->begin() + StartIndex, input->begin() + EndIndex + 1);

		temp_reverse.assign(input->begin() + EndIndex, input->end());
		temp.assign(input->begin(), input->begin() + StartIndex + 1);
		temp_reverse.insert(temp_reverse.end(), temp.begin(), temp.end());
		vector <Point> rightside_1(temp_reverse.rbegin(), temp_reverse.rend());
		rightside.assign(rightside_1.begin(), rightside_1.end());
	}
	if (ht)
	{
		ht = false;
		Point temph;
		vector <Point> vTemp;
		temph = head;
		head = tail;
		tail = temph;
		vTemp = leftside;
		leftside = rightside;
		rightside = vTemp;

	}

	if ((Dist(pre_head, head) > Dist(pre_head, tail) && (pre_head.x + pre_head.y != 0)) || (Dist(head, tail) <averageLength * 21 / 51))
	{

		//cout << "length: " << averageLength << "\n";
		if (Dist(head, tail) < averageLength * 21 /51)
		{

			Mat Orignal_img_bgr, Post_img_bgr;
			cvtColor(Pre_img, Orignal_img_bgr, COLOR_GRAY2BGR);
			cvtColor(Orignal_img, Post_img_bgr, COLOR_GRAY2BGR);

			circle(Orignal_img_bgr, pre_head, 5, Scalar(0, 0, 255), -1);
			circle(Post_img_bgr, head, 5, Scalar(0, 0, 255), -1);

			Mat combined_img;
			hconcat(Orignal_img_bgr, Post_img_bgr, combined_img);
			
			cout << "\n出现自闭行为，头尾有可能识别错误。左侧的红点为前一帧的头部，右侧的红点是当前帧的头部.请判断是否交换?是请输入“y”，否则输入“n” (y/n): ";
			imshow("Combined Image", combined_img);

			waitKey(0); // 等待用户按键以确保图像显示
			char response;
			cin >> response; // 使用 cin 读取用户输入

			if (response == 'y' || response == 'Y')
			{
				Point pTemp;
				vector<Point> vTemp;

				pTemp = head;
				head = tail;
				tail = pTemp;

				vTemp = leftside;
				leftside = rightside;
				rightside = vTemp;
			}

			destroyWindow("Combined Image");
			std::cout << "\n\n正在计算特征点中，请稍等..." << std::endl;
		}
		else
		{
			Point pTemp;
			vector<Point> vTemp;

			pTemp = head;
			head = tail;
			tail = pTemp;

			vTemp = leftside;
			leftside = rightside;
			rightside = vTemp;
		}
	}

	if (leftside[0] != head)
	{
		vector<Point> Temp(leftside.rbegin(), leftside.rend());
		leftside = Temp;
	}

	if (rightside[0] != head)
	{
		vector<Point> Temp(rightside.rbegin(), rightside.rend());
		rightside = Temp;
	}

	free(pt_fa);
	free(pt_ba);
	free(dotproduct_1);
	free(dotproduct_2);
	free(dotproduct);
}
void WormTrack::ResampleByOmit(vector<Point>* input, vector<Point>& output, int pointnum)
{
	float n = input->size() / (float)pointnum;

	for (int i = 0; i < pointnum; i++)
	{
		output.push_back((*input)[(int)(n * i + 0.5)]);
	}
}

int WormTrack::FindCenterline(vector<Point>* inputleft, vector<Point>* inputright, vector<Point>& centerline)
{
	if (inputleft->size() != inputright->size())
	{
		return 0;
	}

	Point pt;
	for (int i = 0; i < inputleft->size(); i++)
	{
		pt.x = (int)(((*inputleft)[i].x + (*inputright)[i].x)*0.5 + 0.5);
		pt.y = (int)(((*inputleft)[i].y + (*inputright)[i].y)*0.5 + 0.5);

		centerline.push_back(pt);
	}
	return 1;
}

int WormTrack::ResampleDist(vector<Point>* input, vector<Point>& output, vector<Point>& lookuptab, int pointnum)
{
	float *dist, totaldist = 0;
	dist = (float *)malloc(input->size() * sizeof(float));
	dist[0] = 0;

	for (int i = 1; i < input->size(); i++)
	{
		totaldist += Dist((*input)[i - 1], (*input)[i]);
		dist[i] = totaldist;
	}

	float averagelength = totaldist / pointnum;
	float length = 0;
	float weightA, weightB;
	Point pt;
	Point lookup;

	int preindex = 0, currindex = 1;

	output.push_back((*input)[0]);
	lookuptab.push_back(Point(0, 0));


	for (int i = 1; i < pointnum; i++)
	{
		length = i * averagelength < totaldist ? i * averagelength : totaldist;

		while (length < dist[preindex] || length > dist[currindex])
		{
			preindex++;
			currindex++;
		}


		weightA = (dist[currindex] - length) / (dist[currindex] - dist[preindex]);
		weightB = 1 - weightA;

		pt.x = (int)(weightA * (*input)[preindex].x + weightB * (*input)[currindex].x + 0.5);
		pt.y = (int)(weightA * (*input)[preindex].y + weightB * (*input)[currindex].y + 0.5);

		output.push_back(pt);


		lookup.x = i;
		lookup.y = weightA > weightB ? preindex : currindex;
		lookuptab.push_back(lookup);
	}

	output.push_back((*input)[input->size() - 1]);
	lookuptab.push_back(Point(pointnum, input->size() - 1));

	free(dist);
	return 1;
}

float WormTrack::Dist(Point a, Point b)
{
	return sqrt(((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y)));
}

float WormTrack::SquareDist(Point a, Point b)
{
	return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y);
}

void WormTrack::SegmentWorm(vector<Point>* ReCenterline, vector<Point>* Centerline, vector<Point>* lookuptab, vector<Point>* BoundA, vector<Point>* BoundB, vector<Point>* Contours, seg* segment)
{
	Point targent;
	Point forward, backward;
	int indexA = 0, indexB = 0;
	Point resultA = (*BoundA)[0], resultB = (*BoundB)[0];



	int	averlength = (int)((float)(BoundA->size() > BoundB->size() ? BoundA->size() : BoundB->size()) / ReCenterline->size() + 0.5);

	int step = 2;

	for (int i = 0; i < ReCenterline->size(); i++)
	{
		backward = (*Centerline)[((*lookuptab)[i].y - step > 0) ? ((*lookuptab)[i].y - step) : 0];
		forward = (*Centerline)[((*lookuptab)[i].y + step < Centerline->size()) ? ((*lookuptab)[i].y + step) : (Centerline->size() - 1)];

		targent = forward - backward;

		indexA = FindPerpPoint(BoundA, (*ReCenterline)[i], targent, resultA, ReCenterline->size() - 1, indexA, indexA + averlength * 2);
		indexB = FindPerpPoint(BoundB, (*ReCenterline)[i], targent, resultB, ReCenterline->size() - 1, indexB, indexB + averlength * 2);

		segment->segAB.push_back(Point(indexA, indexB));
	}
}

void WormTrack::newSegmentWorm(vector<Point>* Centerline, vector<Point>* BoundA, vector<Point>* BoundB, seg* segment)
{
	Point targent;
	Point forward, backward;
	int indexA = 0, indexB = 0;
	Point resultA = (*BoundA)[0], resultB = (*BoundB)[0];

	int averagelength = (int)(((BoundA->size() > BoundB->size()) ? BoundA->size() : BoundB->size()) / partnum + 0.5);

	int step = 2;

	for (int i = 0; i < Centerline->size(); i++)
	{
		backward = (*Centerline)[(i - step > 0) ? (i - step) : 0];
		forward = (*Centerline)[(i + step < Centerline->size()) ? (i + step) : (Centerline->size() - 1)];

		targent = forward - backward;

		indexA = FindPerpPoint(BoundA, (*Centerline)[i], targent, resultA, Centerline->size() - 1, indexA, indexA + averagelength + 2);
		indexB = FindPerpPoint(BoundB, (*Centerline)[i], targent, resultB, Centerline->size() - 1, indexB, indexB + averagelength + 2);

		segment->segAB.push_back(Point(indexA, indexB));
	}
}


int WormTrack::FindPerpPoint(vector<Point>* input, Point x, Point targent, Point &result, int partnum, int startindex, int endindex)
{
	if (abs((*input)[0].x - x.x) + abs((*input)[0].y - x.y) <1)
	{
		result = x;
		return 0;
	}
	else if (abs((*input)[input->size() - 1].x - x.x) + abs((*input)[input->size() - 1].y - x.y) <1)
	{
		result = x;
		return input->size() - 1;
	}

	int mindot = INT_MAX;
	int temp, index = startindex;
	Point pt;

	if (partnum <= 5)
	{
		endindex = endindex + 40;
	}

	endindex = endindex;

	startindex = startindex >= 0 ? startindex : 0;
	startindex = startindex <= input->size() - 1 ? startindex : input->size() - 1;
	endindex = endindex <= input->size() - 1 ? endindex : input->size() - 1;


	for (int i = startindex; i <= endindex; i++)
	{
		pt = (*input)[i] - x;
		temp = pt.x * targent.x + pt.y * targent.y;
		temp = abs(temp);
		if (mindot > temp)
		{
			mindot = temp;
			index = i;
		}
	}

	result = (*input)[index];
	return index;
}

void  WormTrack::newFindCenterline(vector<Point>* BoundA, vector<Point>* BoundB, vector<Point>* centerline, int pointnum, int step)
{

	vector<Point> tabA, newboundA, tabB, newboundB, roughcenterline;
	ResampleDist(BoundA, newboundA, tabA, pointnum);
	Point backward, forward, targent, resultA, resultB, CenterA, CenterB, Center;
	int indexA = 0, indexB = 0, averagelength = (int)(BoundB->size() / pointnum);

	int omit_tail = pointnum / 20;
	int omit_head = pointnum / 20;

	for (int i = omit_head; i < pointnum - omit_tail; i++)
	{
		backward = newboundA[i - step > 0 ? i - step : 0];
		forward = newboundA[i + step < (pointnum - 1) ? i + step : (pointnum - 1)];
		targent = forward - backward;

		if (i == omit_head)
		{
			indexA = FindPerpPoint(BoundB, newboundA[i], targent, resultA, pointnum, indexA, indexA + BoundB->size() / 20);
		}
		else
		{
			indexA = FindPerpPoint(BoundB, newboundA[i], targent, resultA, pointnum, indexA, indexA + averagelength + 5);
		}

		CenterA = Point((int)(0.5*(newboundA[i].x + resultA.x) + 0.5), (int)(0.5*(newboundA[i].y + resultA.y) + 0.5));


		roughcenterline.push_back(CenterA);

	}

	vector<Point> pre_center;

	SmoothSequence(&roughcenterline, pre_center, 1);

	pre_center.insert(pre_center.begin(), newboundA[0]);
	pre_center.push_back(newboundA[newboundA.size() - 1]);

	vector<Point> tabC;
	ResampleDist(&pre_center, *centerline, tabC, partnum);
}

void WormTrack::CompensateOffset(vector<Point>* input, vector<Point>* output, Point topleft)
{
	Point curr;
	for (int i = 0; i < input->size(); i++)
	{
		curr = Point((*input)[i].x + topleft.x, (*input)[i].y + topleft.y);
		output->push_back(curr);
	}

	int botx = 0, boty = 0, topx = Orignal_img.size().width, topy = Orignal_img.size().height;

	for (int i = 0; i < input->size(); i++)
	{
		topx = (*output)[i].x < topx ? (*output)[i].x : topx;
		topy = (*output)[i].y < topy ? (*output)[i].y : topy;
		botx = (*output)[i].x > botx ? (*output)[i].x : botx;
		boty = (*output)[i].y > boty ? (*output)[i].y : boty;
	}

	A = Point(topx, topy);
	B = Point(botx, boty);
}