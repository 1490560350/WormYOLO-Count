#include "stdafx.h"

#ifdef WIN32   
#include <windows.h>
#else         
#include <sys/time.h>
#endif

#include <fstream>

using namespace std;

class myTimer
{
public:
	myTimer(void);
	~myTimer(void);

private:
	LARGE_INTEGER currentCount;

	LARGE_INTEGER startCount;

	LARGE_INTEGER endCount;

	LARGE_INTEGER freq;

public:
	double dbTime;
	double cTime;
	double CurrentTime;
	double totalTime;
public:
	void GetTime();
	void StartTimer(const char* name);
	void StopTimer();
};