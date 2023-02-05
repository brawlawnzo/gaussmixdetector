#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core/mat.hpp>

constexpr double defaultT { 0.1 };
constexpr unsigned int defaultHistory { 100U };
constexpr double defaultDeviation { 40.0 };
constexpr double defaultCf { 0.05 };

class GaussMixDetector
{
	// should figure out which parameters can and cannot be changed 'on-the-fly' and
	// add 'set parameter x' function(s) to allow adjusting the model without reinitializing

	static constexpr uchar K { 3U };                            // limit number of Gaussians per pixel
	double alpha { 1 / static_cast<double>(defaultHistory) };   // learning coefficient
	double T { defaultT };                                      // background-foreground threshold
	double initDeviation { defaultDeviation };                  // initial 2.5*deviation of all Gaussians
	double Cf { defaultCf };                                    // portion of FG data

	int fRows { 0 }, fCols { 0 }, fChannels { 0 };              // frame parameters

	bool firstFrame { true };                                   // first step flag

	std::vector <cv::Mat> mean;
	std::vector <cv::Mat> weight;
	std::vector <cv::Mat> deviation;
	cv::Mat currentK;                                           // current number of Gaussians for each pixel

	static const int CVType = CV_64F;                           // type of 'Mat' pixel info
	typedef double ptrType;                                     // 'Mat' pointer type

public:
	GaussMixDetector() = default;
	explicit GaussMixDetector ( unsigned int _historyLength, double _initDeviation = defaultDeviation, double _T = defaultT, double _Cf = defaultCf );

	void getMotionPicture( const cv::Mat& frame, cv::Mat& motion, bool cleanup = true );

private:
	void Init(const cv::Mat& frame);
	template <typename matPtrType>
	void getpwUpdateAndMotion(const cv::Mat& frame, cv::Mat& motion);
	template <typename matPtrType, int channels>
	void getpwUpdateAndMotionRGB(const cv::Mat& frame, cv::Mat& motion);
};

