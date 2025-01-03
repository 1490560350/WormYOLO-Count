#pragma once
#include "stdafx.h"
#include "getAllFilesTest.h"
#include "Timer.h"
#include "WormTrack.h"
#include <string>
#include <io.h>
#include <iostream>
#include <ctime>
#include <regex>
#include <direct.h>
#include <queue>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


int RefineCurvature(seg* segment, vector<Point>& cur);
int DrawImage(seg& segment, Mat& I, int partnum, vector<Point> cur);
int CurvatureAnalysis(vector<double>* curvature, vector<Point>& cur);
int FindMax(vector<double>* input, int start, int end);
int FindMin(vector<double>* input, int start, int end);
int FindPerpPoint(vector<Point>* input, Point x, Point target, Point &result, int startindex, int endindex);
void Smooth1DSequence(const vector<double>* input, vector<double>& output, double sigma);


Point m_A = Point(0, 0);
Point m_B = Point(0, 0); 
Point head = Point(0, 0);
Point tail = Point(0, 0);
float totaLength = 0;


void CheckBoundary(Point &pt, int maxX, int maxY) {
	pt.x = max(pt.x, 0);
	pt.y = max(pt.y, 0);
	pt.x = min(pt.x, maxX);
	pt.y = min(pt.y, maxY);
}


double calculateAngleDifference(Point p1, Point p2, Point q1, Point q2) {
	double vector1_x = p1.x - p2.x;
	double vector1_y = p1.y - p2.y;
	double vector2_x = q1.x - q2.x;
	double vector2_y = q1.y - q2.y;
	double dotProduct = vector1_x * vector2_x + vector1_y * vector2_y;
	double crossProduct = vector1_x * vector2_y - vector1_y * vector2_x;
	if (std::abs(dotProduct) == 0) dotProduct = 0.0;
	if (std::abs(crossProduct) == 0) crossProduct = 0.0;
	double magnitude1 = sqrt(vector1_x * vector1_x + vector1_y * vector1_y);
	double magnitude2 = sqrt(vector2_x * vector2_x + vector2_y * vector2_y);
	double angleDifference = atan2(crossProduct, dotProduct);

	return abs(angleDifference);
}

void ProcessImage(int j, const string &Path, seg &segment, Point &prevHead, Point &prevPharynx, Point &m_A, Point &m_B, bool switchht, double &angleSumHead, int partnum, float &segmentLength, int scaleimg) {
	string name = Path + to_string(j) + ".png";
	string name1 = Path + to_string(j - 1) + ".png";

	Mat I = imread(name, 0);
	Mat I1 = imread(name1, 0);
	Mat I_color = imread(name, 1);

	if (I.empty()) {
		std::cerr << "无法读取图像: " << name << std::endl;
		return;
	}


	if (m_B == Point(0, 0)) {
		m_B = Point(I.cols, I.rows);
	}

	int guassian = 1;
	int threshold = 70;
	int morpology = 1;
	float segmentLeng = segmentLength;
	WormTrack Track(Path, I, I1, j, scaleimg, guassian, threshold, morpology, partnum, m_A, m_B, &segment, switchht, segmentLeng);

	Track.PreProcess();
	Track.Contour();
	Track.Analysis();
	Track.Curvaturecalculation();
	Track.Lengthcalculation();
	Track.Widthcalculation();
	Track.Volumecalculation();
	Track.Segment();

	segment = Track.segment;

	totaLength += segment.Length;
	std::cout << "Segment Length: " << segment.Length << std::endl;

	segmentLength = totaLength / j;

	double angleDifferenceHead = calculateAngleDifference(segment.head, segment.center[12], prevHead, prevPharynx);
	angleSumHead += angleDifferenceHead;
	angleDifferenceHead = 0.0;

	prevHead = segment.head;
	prevPharynx = segment.center[12];
	int boundary = static_cast<int>(std::floor(segment.Length / 5.0));
	m_A = Track.segment.topleft - Point(boundary, boundary);
	m_B = Track.segment.botright + Point(boundary, boundary);

	int maxX = (I_color.cols - 1);
	int maxY = (I_color.rows - 1);
	CheckBoundary(m_A, maxX, maxY);
	CheckBoundary(m_B, maxX, maxY);
}


int GetFileNum(const std::string& inPath, const std::vector<std::string>& formats)
{
	int fileNum = 0;
	std::vector<std::string> pathVec;
	std::queue<std::string> q;
	q.push(inPath);
	while (!q.empty())
	{
		std::string item = q.front(); q.pop();

		struct _finddata_t fileinfo;
		intptr_t handle = -1;
		for (size_t i = 0; i < formats.size(); ++i) {
			std::string path = item + "\\*" + formats[i];
			handle = _findfirst(path.c_str(), &fileinfo);
			if (handle == -1) continue;

			do {
				if (fileinfo.attrib & _A_SUBDIR)
				{
					if (strcmp(fileinfo.name, ".") == 0 || strcmp(fileinfo.name, "..") == 0) continue;
					q.push(item + "\\" + fileinfo.name);
				}
				else
				{
					fileNum++;
					pathVec.push_back(item + "\\" + fileinfo.name);
				}
			} while (_findnext(handle, &fileinfo) == 0);
			_findclose(handle);
		}
	}
	return fileNum;
}

Point saved_m_A;
Point saved_m_B;

int main() {
	myTimer Timer;
	seg segment;
	float segmentLength = 0;
	Point prevHead = Point(0, 0);
	Point prevPharynx = Point(0, 0);
	segment.head = head;
	segment.tail = tail;
	std::string Path = "samples\\";
	std::string LoadPath = Path + "analysis result\\";

	if (_access(LoadPath.c_str(), 0) == -1) {
		_mkdir(LoadPath.c_str());
	}

	std::ofstream file(LoadPath + "Pharynx.csv");
	std::ofstream file1(LoadPath + "PeakPoints.csv");
	std::ofstream file2(LoadPath + "InflectionPoints.csv");
	std::ofstream file3(LoadPath + "HeadTailReg.csv");
	std::ofstream file4(LoadPath + "SkeletonPoints.csv");
	vector<std::string> files;

	
	std::vector<std::string> formats = { ".png", ".jpg", ".jpeg", ".bmp", ".tiff" };
	int FileNum = GetFileNum(Path, formats);

	double angleSumHead = 0.0;
	double angleSumHead1 = 0.0;
	int partnum = 120;

	int scaleimg = 0;
	{
		
		string firstImagePath = Path + "1.png";
		
		bool found = false;
		for (size_t i = 0; i < formats.size(); ++i) {
			string tempPath = Path + "1" + formats[i];
			Mat I_temp = imread(tempPath, 0);
			if (!I_temp.empty()) {
				scaleimg = 1; // 	
				m_B = Point(I_temp.cols, I_temp.rows);
				found = true;
				break;
			}
		}
		if (!found) {
			std::cerr << "未找到第一张图像。" << std::endl;
			return -1;
		}

		Mat I = imread(Path + to_string(1) + ".png", 0);
		if (I.empty()) {
			std::cerr << "无法读取图像: " << Path + "1.png" << std::endl;
			return -1;
		}
		vector<Point> black_pixels;
		for (int y = 0; y < I.rows; y++) {
			for (int x = 0; x < I.cols; x++) {
				if (I.at<uchar>(y, x) == 0) {
					black_pixels.push_back(Point(x, y));
				}
			}
		}
		if (!black_pixels.empty()) {
			RotatedRect minRect = minAreaRect(black_pixels);
			float minRectArea = minRect.size.width * minRect.size.height;

			float imageArea = I.rows * I.cols;
			float areaRatio = minRectArea / imageArea;

			scaleimg = static_cast<int>(floor(sqrt(0.06 / areaRatio)));
			scaleimg = std::max(scaleimg, 1);
		}
	}

	for (int j = 1; j <= FileNum / 10; ++j) {
		ProcessImage(j, Path, segment, prevHead, prevPharynx, m_A, m_B, false, angleSumHead, partnum, segmentLength, scaleimg);
	}
	std::cout << "angleSumHead的值是: " << angleSumHead << std::endl;
	double angleDifferenceHead = 0.0;
	m_A = Point(0, 0);
	m_B = Point(0, 0); 
	segment.tail = Point(0, 0);
	segment.head = Point(0, 0);
	segment.center[12] = Point(0, 0);
	prevHead = Point(0, 0);
	prevPharynx = Point(0, 0);
	segmentLength = 0;
	totaLength = 0;

	for (int j = 1; j <= FileNum / 10; ++j) {
		ProcessImage(j, Path, segment, prevHead, prevPharynx, m_A, m_B, true, angleSumHead1, partnum, segmentLength, scaleimg);

		if (j == 1) {
			saved_m_A = m_A;
			saved_m_B = m_B;
		}
	}
	std::cout << "angleSumHead1的值是: " << angleSumHead1 << std::endl;
	bool switchht = (angleSumHead > angleSumHead1) ? false : true;
	std::cout << "switchht: " << switchht << std::endl;
	totaLength = 0;
	int scaleimg1 = static_cast<int>(ceil(200.0f / segmentLength));

	scaleimg = std::min(scaleimg, scaleimg1);
	m_A = saved_m_A;
	m_B = saved_m_B;
	segment.tail = Point(0, 0);
	segment.head = Point(0, 0);

	for (int j = 1; j <= FileNum + 1; ++j) {
		bool currentSwitchht = (j == 1 && switchht == true) ? true : false;
		ProcessImage(j, Path, segment, prevHead, prevPharynx, m_A, m_B, switchht, angleSumHead, partnum, segmentLength, scaleimg);
		prevHead = segment.head;
		std::cout << "平均长度: " << segmentLength << std::endl;
		Mat I_color = imread(Path + to_string(j) + ".png", 1);
		if (I_color.empty()) {
			std::cerr << "无法读取图像: " << Path + to_string(j) + ".png" << std::endl;
			continue;
		}
		vector<Point> cur;
		RefineCurvature(&segment, cur);

		if (file) {
			file << segment.center[12].x << "," << segment.center[12].y;
			file3 << segment.head.x << "," << segment.head.y << "," << segment.tail.x << "," << segment.tail.y;
		}
		if (file4) {
			for (size_t i = 0; i < segment.center.size(); ++i) {
				file4 << segment.center[i].x << "," << segment.center[i].y << ",";
			}
		}
		for (size_t i = 0; i < cur.size(); i++) {
			if (cur[i].x == 0) {
				if (file2) {
					file2 << segment.center[cur[i].y].x << "," << segment.center[cur[i].y].y << ",";
				}
			}
			else if (cur[i].x == -1) {
				if (file1) {
					file1 << segment.center[cur[i].y].x << "," << segment.center[cur[i].y].y << ",";
				}
			}
			else {
				if (file1) {
					file1 << segment.center[cur[i].y].x << "," << segment.center[cur[i].y].y << ",";
				}
			}
		}

		if (file) {
			file << "\n";
		}
		if (file1) {
			file1 << "\n";
		}
		if (file2) {
			file2 << "\n";
		}
		if (file3) {
			file3 << "\n";
		}
		if (file4) {
			file4 << "\n";
		}
	}

	file.close();
	file1.close();
	file2.close();
	file3.close();
	file4.close();
	waitKey(0);
	return 0;
}



int RefineCurvature(seg* segment, vector<Point>& result)
{
	if (segment->curvatureC.size() == 0)
	{
		return 0;
	}

	vector<Point> curC;
	vector<double> curvatureA, curvatureB, scurvA, scurvB;
	CurvatureAnalysis(&(segment->curvatureC), curC);

	if (curC.size() < 5)
	{
		return 0;
	}

	curvatureA.insert(curvatureA.begin(), (segment->curvatureA).begin() + 1, (segment->curvatureA).end() - 1);
	curvatureB.insert(curvatureB.begin(), (segment->curvatureB).begin() + 1, (segment->curvatureB).end() - 1);

	Smooth1DSequence(&(curvatureA), scurvA, 3);
	Smooth1DSequence(&(curvatureB), scurvB, 3);


	int a, b;
	a = 0;

	int length = ((segment->Partnum / curC.size()) > 1 ? (segment->Partnum / curC.size()) : 1) * 1.5;

	Point backward, forward, tangent;
	int step = 10;
	for (int i = 0; i < curC.size(); i++)
	{
		b = (a + length) < (segment->center.size() - 1) ? (a + length) : (segment->center.size() - 1);

		if (curC[i].x == 0)
		{
			result.push_back(curC[i]);
		}
		else if (curC[i].x == 1)
		{
			int start = (curC[i].y - length) > 0 ? (curC[i].y - length) : 0;
			int end = (curC[i].y + length) < scurvA.size() ? (curC[i].y + length) : scurvA.size();

			Point perp;
			int index = segment->tabA[1 + FindMax(&scurvA, start, end)].y;

			backward = segment->contourA[(index - step) > 0 ? (index - step) : 0];
			forward = segment->contourA[(index + step) < (segment->contourA.size() - 1) ? (index + step) : (segment->contourA.size() - 1)];
			tangent = forward - backward;

			result.push_back(Point(1, FindPerpPoint(&(segment->center), segment->contourA[index], tangent, perp, a, b)));
		}
		else
		{
			int start = (curC[i].y - length) > 0 ? (curC[i].y - length) : 0;
			int end = (curC[i].y + length) < scurvB.size() ? (curC[i].y + length) : scurvB.size();

			Point perp;
			int index = segment->tabB[1 + FindMin(&scurvB, start, end)].y;

			backward = segment->contourB[(index - step) > 0 ? (index - step) : 0];
			forward = segment->contourB[(index + step) < (segment->contourB.size() - 1) ? (index + step) : (segment->contourB.size() - 1)];
			tangent = forward - backward;

			result.push_back(Point(-1, FindPerpPoint(&(segment->center), segment->contourB[index], tangent, perp, a, b)));
		}


		a = result[i].y + length * 0.3;
	}

	return 1;
}

int FindPerpPoint(vector<Point>* input, Point x, Point targent, Point &result, int startindex, int endindex)
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

int FindMax(vector<double>* input, int start, int end)
{
	double Max = -100;
	int indexMax = start;

	for (int i = start; i < end; i++)
	{
		if ((*input)[i] > Max)
		{
			indexMax = i;
			Max = (*input)[i];
		}
	}
	return indexMax;
}

int FindMin(vector<double>* input, int start, int end)
{
	double Min = 100;
	int indexMin = start;

	for (int i = start; i < end; i++)
	{
		if ((*input)[i] < Min)
		{
			indexMin = i;
			Min = (*input)[i];
		}
	}
	return indexMin;
}

int CurvatureAnalysis(vector<double>* curvature, vector<Point>& cur)
{
	if (curvature->size() == 0)
	{
		return 0;
	}


	vector<double> curv, scurv, super_scurv;
	curv.insert(curv.begin(), curvature->begin() + 1, curvature->end() - 1);

	Smooth1DSequence(&curv, scurv, 1);
	Smooth1DSequence(&curv, super_scurv, 3);

	float omit = 0.08;
	for (int i = 0; i < curv.size()*omit; i++)
	{
		scurv[i] = 0;
		super_scurv[i] = 0;
	}
	for (int i = curv.size()*(1 - omit); i < curv.size(); i++)
	{
		scurv[i] = 0;
		super_scurv[i] = 0;
	}


	vector<int> pre_Zeros, Zeros;
	vector<int>	pre_Max, pre_Min;
	Point Headpeak = Point(0, 0);
	Point Tailpeak = Point(0, 0);


	for (int i = 0; i < scurv.size() - 1; i++)
	{
		if (scurv[i] == 0 || scurv[i + 1] * scurv[i] <0)
		{
			pre_Zeros.push_back(i);

		}

	}


	for (int i = 0; i < pre_Zeros.size() - 1; i++)
	{
		if (pre_Zeros[i + 1] - pre_Zeros[i] > curv.size() / 6 && scurv[pre_Zeros[i] + 1] > 0)
		{
			double Max = 0;
			int indexMax = pre_Zeros[i] + 1;
			for (int j = pre_Zeros[i] + 1; j < pre_Zeros[i + 1] - 1; j++)
			{
				if (Max < super_scurv[j])
				{
					Max = super_scurv[j];
					indexMax = j;
				}
			}
			pre_Max.push_back(indexMax);


			if (Zeros.size() == 0)
			{
				Zeros.push_back(pre_Zeros[i]);
				Zeros.push_back(pre_Zeros[i + 1]);
			}
			else
			{
				if (Zeros[Zeros.size() - 1] != pre_Zeros[i])
				{
					Zeros.push_back(pre_Zeros[i]);
					Zeros.push_back(pre_Zeros[i + 1]);
				}
				else
				{
					Zeros.push_back(pre_Zeros[i + 1]);
				}
			}

		}

		if (pre_Zeros[i + 1] - pre_Zeros[i] > curv.size() / 6 && scurv[pre_Zeros[i] + 1] < 0)
		{
			double Min = 100;
			int indexMin = pre_Zeros[i] + 1;
			for (int j = pre_Zeros[i] + 1; j < pre_Zeros[i + 1] - 1; j++)
			{
				if (Min > super_scurv[j])
				{
					Min = super_scurv[j];
					indexMin = j;
				}
			}
			pre_Min.push_back(indexMin);

			if (Zeros.size() == 0)
			{
				Zeros.push_back(pre_Zeros[i]);
				Zeros.push_back(pre_Zeros[i + 1]);
			}
			else
			{
				if (Zeros[Zeros.size() - 1] != pre_Zeros[i])
				{
					Zeros.push_back(pre_Zeros[i]);
					Zeros.push_back(pre_Zeros[i + 1]);
				}
				else
				{
					Zeros.push_back(pre_Zeros[i + 1]);
				}
			}

		}
	}




	if (Zeros.size() != 0)
	{
		if (Zeros[0]>0)
		{
			if (scurv[Zeros[0] - 1] >0)
			{
				double Max = 0;
				int indexMax = 1;
				for (int j = 1; j < Zeros[0] - 1; j++)
				{
					if (Max < super_scurv[j])
					{
						Max = super_scurv[j];
						indexMax = j;
					}
				}
				Headpeak = Point(1, indexMax);
			}
			else
			{
				double Min = 100;
				int indexMin = 0;
				for (int j = 1; j < Zeros[0] - 1; j++)
				{
					if (Min > super_scurv[j])
					{
						Min = super_scurv[j];
						indexMin = j;
					}
				}
				Headpeak = Point(-1, indexMin);
			}
		}
	}


	if (Zeros.size() != 0)
	{
		if (Zeros[Zeros.size() - 1] < scurv.size() - 1)
		{
			if (scurv[Zeros[Zeros.size() - 1] + 1] >0)
			{
				double Max = 0;
				int indexMax = Zeros[Zeros.size() - 1] + 1;
				for (int j = Zeros[Zeros.size() - 1] + 1; j < scurv.size() - 2; j++)
				{
					if (Max < super_scurv[j])
					{
						Max = super_scurv[j];
						indexMax = j;
					}
				}
				Tailpeak = Point(1, indexMax);
			}
			else
			{
				double Min = 100;
				int indexMin = Zeros[Zeros.size() - 1] + 1;
				for (int j = Zeros[Zeros.size() - 1] + 1; j < scurv.size() - 2; j++)
				{
					if (Min > super_scurv[j])
					{
						Min = super_scurv[j];
						indexMin = j;
					}
				}
				Tailpeak = Point(-1, indexMin);
			}

		}
	}


	if (Headpeak != Point(0, 0) && Zeros.size() != 0)
	{
		if (Zeros[0] - Headpeak.y > scurv.size() / 8)
		{
			if (Headpeak.x > 0)
			{
				pre_Max.insert(pre_Max.begin(), Headpeak.y);
			}
			else
			{
				pre_Min.insert(pre_Min.begin(), Headpeak.y);
			}
		}
	}

	if (Tailpeak != Point(0, 0) && Zeros.size() != 0)
	{
		if (Tailpeak.y - Zeros[Zeros.size() - 1] > scurv.size() / 8)
		{
			if (Tailpeak.x > 0)
			{
				pre_Max.push_back(Tailpeak.y);
			}
			else
			{
				pre_Min.push_back(Tailpeak.y);
			}
		}
	}

	for (int i = 0; i < pre_Max.size(); i++)
	{
		cur.push_back(Point(1, pre_Max[i] + 1));
	}
	for (int i = 0; i < pre_Min.size(); i++)
	{
		cur.push_back(Point(-1, pre_Min[i] + 1));
	}
	for (int i = 0; i < Zeros.size(); i++)
	{
		cur.push_back(Point(0, Zeros[i] + 1));
	}

	if (cur.size() != 0)
	{
		for (int i = 0; i < cur.size() - 1; i++)
		{
			for (int j = 0; j < cur.size() - i - 1; j++)
			{
				if (cur[j].y > cur[j + 1].y)
				{
					Point temp = cur[j];
					cur[j] = cur[j + 1];
					cur[j + 1] = temp;
				}
			}
		}
	}

	return 1;
}

void Smooth1DSequence(const vector<double>* input, vector<double>& output, double sigma)
{
	int *kernel, klength, normfactor;

	int ll, ul, x;
	double n;
	ll = (int)(-3 * sigma) - 1;
	ul = (int)(3 * sigma) + 1;
	klength = ul - ll + 1;
	kernel = (int*)malloc(klength * sizeof(int));

	normfactor = 0;
	n = exp(-1.0*ll*ll / (2 * sigma*sigma));
	for (x = 0; x < klength; x++)
	{
		(kernel)[x] = (int)(exp(-1.0*(x + ll)*(x + ll) / (2 * sigma*sigma)) / n + 0.5);
		normfactor += (kernel)[x];
	}


	int j, k, ind, anchor;
	double sum;

	double *pt;
	int length = input->size();
	pt = (double*)malloc(length * sizeof(Point));
	anchor = klength / 2;
	for (j = 0; j < length; j++)
	{
		sum = 0;
		for (k = 0; k < klength; k++)
		{
			ind = j + k - anchor;
			ind = ind > 0 ? ind : 0;
			ind = ind < length ? ind : (length - 1);
			sum = sum + (*input)[ind] * kernel[k];
		}

		pt[j] = (double)(1.0*sum / normfactor);

	}
	output = vector<double>(pt, pt + length);

	free(pt);

}


