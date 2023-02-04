#include "GaussMixDetector.h"

#include <array>
#include <cassert>

#include <opencv2/imgproc.hpp>


GaussMixDetector::GaussMixDetector( unsigned int _historyLength, double _initDeviation, double _T, double _Cf )
	: alpha { 1 / static_cast<double>(_historyLength) }
	, T { _T }
	, initDeviation { _initDeviation }
	, Cf { _Cf }
{
}

void GaussMixDetector::Init( const cv::Mat& frame )
{
	assert(!frame.empty());
	if (frame.rows < 1 || frame.cols < 1)
	{
		throw std::invalid_argument("Image has invalid size.");
	}

	cv::Mat tmp;
	fRows = frame.rows;
	fCols = frame.cols;
	fChannels = frame.channels();

	frame.convertTo( tmp, CV_MAKETYPE( CVType, fChannels ) );

	mean.push_back( tmp );
	weight.emplace_back(fRows, fCols, CV_MAKETYPE( CVType, 1 ), cv::Scalar( 1 ));

	for ( int k = 1; k < K; k++ )
	{
		mean.emplace_back(fRows, fCols, CV_MAKETYPE( CVType, fChannels ), cv::Scalar( 0, 0 ));
		weight.emplace_back(fRows, fCols, CV_MAKETYPE( CVType, 1 ), cv::Scalar( 0 ));
	}

	tmp.create( fRows, fCols, CV_MAKETYPE( CVType, fChannels*fChannels ) );
	for ( int k = 0; k < K; k++ )
	{
		for( int r = 0; r < fRows; r++ )
		{
			auto* p = tmp.ptr<double>(r);
			for( int c = 0; c < fCols*fChannels*fChannels; c++ )
			{
				p[c] = ( (c % (fChannels*fChannels)) % (fChannels+1) == 0 ) ? initDeviation : 0;
			}
		}
		deviation.push_back( tmp );
	}

	currentK = cv::Mat( fRows, fCols, CV_MAKETYPE( CV_8U, 1 )
			, cv::Scalar( 1 ) );

	firstFrame = false;
}

inline double normDistrib( double x, double m, double d )
{
	double tmp = (x - m) / d;
	tmp *= tmp;
	tmp = exp( -tmp/2 );
	return tmp / d / sqrt(2*CV_PI);
}

void GaussMixDetector::getpwUpdateAndMotion( cv::Mat& motion )
{
	std::vector<ptrType*> ptM(K);
	std::vector<ptrType*> ptD(K);
	std::vector<ptrType*> ptW(K);
	std::vector<bool> isCurrent(K,false);

	std::array<double, K> tmpM {};
	std::array<double, K> tmpD {};
	std::array<double, K> tmpW {};

	for( int i = 0; i < fRows; i++ )
	{
		const auto* ptF = fClone.ptr<double>(i);
		auto* ptMo = motion.ptr<uchar>(i);
		auto* ptK = currentK.ptr<uchar>(i);
		for ( uchar k = 0U; k < K; k++ )
		{
			ptM[k] = mean[k].ptr<double>(i);
			ptD[k] = deviation[k].ptr<double>(i);
			ptW[k] = weight[k].ptr<double>(i);
		}
		for ( int j = 0; j < fCols*fChannels; j += fChannels )
		{
			double tmpF = ptF[j];
			uchar tmpK = ptK[j];
			for ( uchar k = 0U; k < tmpK; k++ )
			{
				tmpM.at(k) = ptM[k][j];
				tmpD.at(k) = ptD[k][j];
				tmpW.at(k) = ptW[k][j];
			}

			uchar count = 0U;
			for ( uchar k = 0U; k < tmpK; k++ )
			{
				isCurrent[k] = false;
				if (( tmpF < tmpM.at(k) + tmpD.at(k) ) && ( tmpF > tmpM.at(k) - tmpD.at(k) ))
				{
					count++;
					isCurrent[k] = true;
				}
				else
				{
					isCurrent[k] = false;
				}
			}
			if( count == 0U )
			{
				if ( tmpK < K )
				{
					tmpM.at(tmpK) = tmpF;
					tmpD.at(tmpK) = initDeviation;
					tmpW.at(tmpK) = 0;
					isCurrent[tmpK] = true;
					tmpK++;
				}
				else
				{
					tmpM.at(0) = tmpF;
					tmpD.at(0) = initDeviation;
					tmpW.at(0) = 0;
					isCurrent[0] = true;
				}
			}
			else
			{
				for ( uchar k = 0U; k < tmpK; k++ )
				{
					if ( isCurrent[k] ) {
						double r = alpha * normDistrib( tmpF, tmpM.at(k), tmpD.at(k) );
						tmpD.at(k) = sqrt ( (1-r) / 6.25 * tmpD.at(k) * tmpD.at(k) + r * (tmpF - tmpM.at(k)) );
						tmpM.at(k) = (1-r) * tmpM.at(k) + r*tmpF;
					}

					if ( isCurrent[k] ) {
						tmpM.at(k) = (1-alpha) * tmpM.at(k) + alpha * tmpF;
						tmpD.at(k) = sqrt ( (1-alpha) * tmpD.at(k) * tmpD.at(k) / 6.25
							+ alpha * (tmpF - tmpM.at(k)) * (tmpF - tmpM.at(k)) );
					}
				}
			}

			double w = 0;
			for ( uchar k = 0U; k < tmpK; k++ )
			{
				tmpW.at(k) = tmpW.at(k) * (1-alpha) + alpha*int(isCurrent[k]);
				w += tmpW.at(k);
			}

			for ( uchar k = 0U; k < tmpK; k++ )
			{
				tmpW.at(k) = tmpW.at(k) / w;
			}

			w = 0;
			for ( uchar k = 0U; k < tmpK; k++ )
			{
				w += tmpW.at(k);
			}

			assert( w == 1.0 );

			bool noMov = false;
			while (!noMov)
			{
				noMov = true;
				for ( uchar k = 0U; k < tmpK-1U; k++ )
				{
					if ( tmpW.at(k) > tmpW.at(k+1U) )
					{
						std::swap(tmpW.at(k), tmpW.at(k+1U));
						std::swap(tmpM.at(k), tmpM.at(k+1U));
						std::swap(tmpD.at(k), tmpD.at(k+1U));

						noMov = false;
					}
				}
			}

			int motionI = j / fChannels;

			uchar count2 = 0U;
			w = tmpW.at(0);

			for ( count2 = 1U; count2 < tmpK; count2++ )
			{
				if ( w >= Cf ) {
					break;
				}

				w += tmpW.at(count2);
			}

			for ( uchar k = 0U; k < count2; k++ )
			{
				if(isCurrent[k])
				{
					ptMo[motionI] = 255U;
				}
			}


			if( tmpK > 1U )
			{
				bool isMotion = true;
				for ( uchar k = 1U; k < tmpK; k++ )
				{
					if( !(abs(tmpM.at(0) - tmpM.at(k)) > T) )
					{
						isMotion = false;
						break;
					}
				}

				if( isMotion )
				{
					ptMo[motionI] = 255U;
				}
			}

			ptK[j] = tmpK;
			for ( uchar k = 0U; k < tmpK; k++ )
			{
				ptM[k][j] = tmpM.at(k);
				ptD[k][j] = tmpD.at(k);
				ptW[k][j] = tmpW.at(k);
			}
		}
	}
}

template <int channels>
double Mahalanobis(const cv::Matx<double, 1, channels>& x, const cv::Matx<double, channels, channels>& C)
{
	return (x * C.inv()).dot(x);
}

template <>
double Mahalanobis<2>(const cv::Matx12d& x, const cv::Matx22d& C)
{
	// Cholesky decomposition
	std::array<double, 3> L{};
	L.at(0) = C(0, 0);
	L.at(1) = C(1, 0) / L.at(0);
	L.at(2) = C(1, 1) - L.at(1) * C(1, 0);

	// Mahalanobis vector
	double y = x(1) - x(0) * L.at(1);
	y *= y;
	y /= L.at(2);
	y += x(0) * x(0) / L.at(0);

	return y;
}

template <>
double Mahalanobis<3>(const cv::Matx13d& x, const cv::Matx33d& C)
{
	// Cholesky decomposition
	std::array<double, 6> L{};
	L.at(0) = C(0, 0);
	L.at(1) = C(1, 0) / L.at(0);
	L.at(2) = C(1, 1) - L.at(1) * C(1, 0);
	L.at(3) = C(2, 0) / L.at(0);
	L.at(4) = (C(2, 1) - L.at(3) * C(1, 0)) / L.at(2);
	L.at(5) = C(2, 2) - L.at(3) * C(2, 0) - L.at(4) * L.at(4) * L.at(2);

	// Mahalanobis vector
	std::array<double, 3> y;
	y.at(0) = x(0) * x(0) / L.at(0);
	y.at(1) = x(1) - x(0) * L.at(1);
	y.at(1) *= y.at(1);
	y.at(1) /= L.at(2);
	y.at(2) = x(2) - L.at(3) * x(0) - L.at(4) * (x(1) - x(0) * L.at(1));
	y.at(2) *= y.at(2);
	y.at(2) /= L.at(5);

	y.at(0) += y.at(1);
	y.at(0) += y.at(2);

	return y.at(0);
}

void GaussMixDetector::getpwUpdateAndMotionRGB( cv::Mat& motion )
{
	std::array<ptrType*, K> ptM {};
	std::array<ptrType*, K> ptD {};
	std::array<ptrType*, K> ptW {};

	std::array<bool, K> isCurrent {};

	cv::Matx13d tmpF;
	std::array<cv::Matx13d, K> delta {};
	std::array<cv::Matx13d, K> tmpM {};
	std::array<cv::Matx33d, K> tmpD {};
	cv::Matx33d dhelp;

	std::array<double, K> tmpW {};

	for( int i = 0; i < fRows; i++ )
	{
		const auto* ptF = fClone.ptr<double>(i);
		auto* ptMo = motion.ptr<uchar>(i);
		auto* ptK = currentK.ptr<uchar>(i);
		for ( uchar k = 0U; k < K; k++ )
		{
			ptM.at(k) = mean[k].ptr<double>(i);
			ptD.at(k) = deviation[k].ptr<double>(i);
			ptW.at(k) = weight[k].ptr<double>(i);
		}

		uchar tmpK = 0U;
		for ( int j = 0; j < fCols; j++ )
		{
			const int iRGB = j*fChannels;
			const int iDev = j*fChannels*fChannels;
			// !!! why 3 ??
			for ( int c = 0; c < 3; c++ )
			{
				tmpF(c) = ptF[iRGB + c];
			}
			tmpK = ptK[j];

			for ( uchar k = 0U; k < tmpK; k++ )
			{
				for ( int c = 0; c < 3; c++ )
				{
					tmpM.at(k)(c) = ptM.at(k)[iRGB + c];
					for ( int cd = 0; cd < 3; cd++ )
					{
						tmpD.at(k)(c,cd) = ptD.at(k)[iDev + c*fChannels + cd];
					}
					tmpW.at(k) = ptW.at(k)[j];
				}
				delta.at(k) = tmpF - tmpM.at(k);
			}

			uchar count = 0U;
			for ( uchar k = 0U; k < tmpK; k++ )
			{
				isCurrent.at(k) = false;
				if ( Mahalanobis<3>(delta.at(k), tmpD.at(k)) < sqrt( cv::trace(tmpD.at(k)) ) )
				{
					count++;
					isCurrent.at(k) = true;
				}
				else
				{
					isCurrent.at(k) = false;
				}
			}

			if( count == 0U )
			{
				if ( tmpK < K )
				{
					tmpM.at(tmpK) = tmpF;
					tmpD.at(tmpK) = initDeviation*cv::Matx33d::eye();
					tmpW.at(tmpK) = alpha;
					tmpK++;
				}
				else
				{
					tmpM.at(K-1U) = tmpF;
					tmpD.at(K-1U) = initDeviation*cv::Matx33d::eye();
					tmpW.at(K-1U) = alpha;
				}
			}
			else
			{
				for ( uchar k = 0U; k < tmpK; k++ )
				{
					if ( isCurrent.at(k) )
					{
						const double w = (alpha / tmpW.at(k));
						tmpM.at(k) += w * delta.at(k);
						tmpD.at(k) += std::min( 20*alpha, w ) * ( delta.at(k).t()*delta.at(k) );
					}
				}
			}

			{
				double w = 0;
				for ( uchar k = 0U; k < tmpK; k++ )
				{
					tmpW.at(k) = tmpW.at(k) * (1-alpha) + alpha*int(isCurrent.at(k));
					w += tmpW.at(k);
				}

				for ( uchar k = 0U; k < tmpK; k++ )
				{
					tmpW.at(k) = tmpW.at(k) / w;
				}
			}

			bool noMov = false;
			while (!noMov)
			{
				noMov = true;
				for ( uchar k = 0U; k < tmpK-1U; k++ )
				{
					if ( tmpW.at(k) < tmpW.at(k+1U) )
					{
						std::swap(tmpW.at(k), tmpW.at(k+1U));
						cv::swap(tmpM.at(k), tmpM.at(k+1U));
						cv::swap(tmpD.at(k), tmpD.at(k+1U));

						noMov = false;
					}
				}
			}

			bool FG = sqrt(delta[0].dot(delta[0])) > T;

			if( FG )
			{
				ptMo[j] = 255U;
			}

			ptK[j] = tmpK;
			for ( uchar k = 0U; k < tmpK; k++ )
			{
				for ( int c = 0; c < fChannels; c++ )
				{
					ptM.at(k)[iRGB + c] = tmpM.at(k)(c);
					for ( int cd = 0; cd < fChannels; cd++ )
					{
						ptD.at(k)[iDev + c*fChannels + cd] = tmpD.at(k)(c,cd);
					}
					ptW.at(k)[j] = tmpW.at(k);
				}
			}
		}
	}
}

void GaussMixDetector::getMotionPicture( const cv::Mat& frame, cv::Mat& motion, bool cleanup )
{
	if (frame.empty())
	{
		throw std::invalid_argument("No input image.");
	}

	if (firstFrame)
	{
		Init(frame);
		motion = cv::Mat( fRows, fCols, CV_MAKETYPE( CV_8U, 1 ), cv::Scalar( 0 ) );
		return;
	}

	if (fRows != frame.rows || fCols != frame.cols)
	{
		throw std::invalid_argument("Input image size different from initial. Stream must be uniform.");
	}
	if (fChannels != frame.channels())
	{
		throw std::invalid_argument("Input image channels different from initial. Stream must be uniform.");
	}

	motion = cv::Mat( fRows, fCols, CV_MAKETYPE( CV_8U, 1 ), cv::Scalar( 0 ) );
	frame.convertTo( fClone, CV_MAKETYPE(CVType, fChannels) );

	// shouldn't algorithm be universal ??
	// i.e. you should be able to create the model for R and G channels only
	if (fChannels == 1)
	{
		getpwUpdateAndMotion(motion);
	}
	else if (fChannels == 3)
	{
		getpwUpdateAndMotionRGB(motion);
	}
	else
	{
		throw std::invalid_argument("Input image has non-standard number of channels.");
	}

	// optional erode + dilate processing of the motion image
	if (cleanup)
	{
		// mb use const static cv::Mat smCircle = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5));
		const static cv::Mat smCircle = (cv::Mat_<uchar>(5, 5) << 0, 0, 1, 0, 0,
																  0, 1, 1, 1, 0,
																  1, 1, 1, 1, 1,
																  0, 1, 1, 1, 0,
																  0, 0, 1, 0, 0);
		cv::erode(motion, motion, smCircle);
		cv::dilate(motion, motion, smCircle);
	}
}
