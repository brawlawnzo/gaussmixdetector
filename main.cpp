#include <cassert>
#include <iostream>

#include "GaussMixDetector.h"

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>


int main(int argc, char** argv)
{
	const cv::String keys =
		"{help h usage ? |      | print this message    }"
		"{@path          |      | input video file path }"
		"{history hist   |100   | history parameter     }"
		"{dev d          |20.0  | initial deviation     }"
		"{T t            |30.0  | BF threshold          }"
		"{Cf cf          |0.1   | portion of FG data    }"
		"{show preview   |      | enable preview        }";

	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("Gauss Mixture Motion Detector\n");

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	// default name for output video from camera stream
	cv::String inputName("cam");		// should add current date and time to this
	cv::String rawName(inputName);
	bool camera = true;

	if (parser.has("@path"))
	{
		camera = false;
		inputName = parser.get<cv::String>("@path");
	}

	const auto history = parser.get<int>("history");
	const auto deviation = parser.get<double>("dev");
	const auto T = parser.get<double>("T");
	const auto Cf = parser.get<double>("Cf");
	const bool showPreview { parser.has("show") };

	if (!parser.check())
	{
		parser.printErrors();
		return 1;
	}

	GaussMixDetector bg(history, deviation, T, Cf);

	cv::Mat frame;
	cv::Mat motion;

	cv::VideoCapture cap;
	if (camera)
	{
		// try a couple of camera indexes
		// (there's no solid method for finding cameras in OpenCV as of now)
		if ( !cap.open(0) )
		{
			if ( !cap.open(1) )
			{
				cap.open(700);
			}
		}
	}
	else
	{
		cap.open(inputName);
		cap.set(cv::CAP_PROP_CONVERT_RGB, static_cast<double>(false));
		cap.set(cv::CAP_PROP_POS_FRAMES, 0);
	}
	if (!cap.isOpened())
	{
		std::cerr << "Failed to initiate video capture.";
		return 1;
	}

	// if videofile opened it means it has a valid extension
	if (!camera)
	{
		size_t lastdot = inputName.find_last_of('.');
		assert(lastdot != std::string::npos);
		rawName = inputName.substr(0, lastdot);     // cut off the extension
	}

	if (showPreview)
	{
		cv::namedWindow("Frame");
		cv::namedWindow("Motion");
	}

	cv::VideoWriter vidmotion;
	cv::String videoExt(".mp4");

	unsigned int total = 0;

#ifdef LOG_TIME_ALL
	LARGE_INTEGER frequency;        // ticks per second
	LARGE_INTEGER t1, t2;           // ticks
	double elapsedTimeGauss = 0;
#endif

	for (; ; )
	{
		cap >> frame;							// read the source
		if (frame.empty())
		{
			break;				// check if reached the end
		}

#ifdef LOG_TIME_ALL
		QueryPerformanceFrequency(&frequency);
		QueryPerformanceCounter(&t1);
#endif

		bg.getMotionPicture(frame, motion);		// get motion data from detector

#ifdef LOG_TIME_ALL
		QueryPerformanceCounter(&t2);
		t1.QuadPart *= 1000;
		t2.QuadPart *= 1000;
		elapsedTimeGauss += (t2.QuadPart - t1.QuadPart) / frequency.QuadPart;
#endif

		if (showPreview)
		{
			cv::imshow("Motion", motion);
		}

		// Writing motion video
		if (!vidmotion.isOpened())
		{
			vidmotion.open(std::string(rawName).append("_motion").append(videoExt),
				cv::VideoWriter::fourcc('M', 'P', '4', 'V'), camera ? 20 : cap.get(cv::CAP_PROP_FPS),
				cv::Size(frame.cols, frame.rows), false);
		}
		vidmotion << motion;

		if (showPreview)
		{
			cv::imshow("Frame", frame);
		}

		total++;

		if (cv::waitKey(15) == 32)
		{
			break;  // space key to stop the loop
		}
	}
#ifdef LOG_TIME_ALL
	elapsedTimeGauss /= double(total);
#endif

	return 0;
}
