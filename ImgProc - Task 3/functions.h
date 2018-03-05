#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <complex>

using namespace cv;
using namespace std;

/*
* Source: https://homepages.inf.ed.ac.uk/rbf/HIPR2/fourier.htm
*/

// struct for storing fourier transformed images
typedef struct{

	int width;
	int height;
	vector<complex<double>> bValues;
	vector<complex<double>> gValues;
	vector<complex<double>> rValues;

} FourierImage;

// term inside the double sum of discrete fourier transformation
complex<double> sumTerm(unsigned char component, int i, int j, int k, int l, int N, int M) {

	double exponent = 2.0 * CV_PI * ((k * i * 1.0 / N) + (l * j * 1.0 / M));

	complex<double> val = complex<double>(component, 0) * exp(-1i * exponent);

	return val;

}

// discrete fourier transformation
void fourierTransformation(const Mat& src, FourierImage& dst) {

	int width = src.cols;
	int height = src.rows;

	if (width % 2 == 1) width--;
	if (height % 2 == 1) height--;

	for (int k = 0; k < height; k++) {
		for (int l = 0; l < width; l++) {

			complex<double> sumB(0.0, 0.0);
			complex<double> sumG(0.0, 0.0);
			complex<double> sumR(0.0, 0.0);

			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {

					Vec3b color = src.at<Vec3b>(i, j);

					complex<double> valB = sumTerm(color[0], i, j, k, l, height, width);
					complex<double> valG = sumTerm(color[1], i, j, k, l, height, width);
					complex<double> valR = sumTerm(color[2], i, j, k, l, height, width);

					sumB += valB;
					sumG += valG;
					sumR += valR;

				}
			}

			dst.bValues.push_back(sumB);
			dst.gValues.push_back(sumG);
			dst.rValues.push_back(sumR);

		}
	}

	dst.width = width;
	dst.height = height;

}

// term inside the double sum of inverse discrete fourier transformation
complex<double> inverseSumTerm(complex<double> component, int a, int b, int k, int l, int N, int M) {

	double exponent = 2.0 * CV_PI * ((k * a * 1.0 / N) + (l * b * 1.0 / M));

	complex<double> val = component * exp(1i * exponent);

	return val;

}

// inverse discrete fourier transformation
void inverseFourierTransformation(const FourierImage& src, Mat& dst) {

	int width = src.width;
	int height = src.height;

	dst = Mat(Size(width, height), CV_8UC3);

	for (int a = 0; a < height; a++) {
		for (int b = 0; b < width; b++) {

			complex<double> sumB(0.0, 0.0);
			complex<double> sumG(0.0, 0.0);
			complex<double> sumR(0.0, 0.0);

			for (int k = 0; k < height; k++) {
				for (int l = 0; l < width; l++) {

					complex<double> valB = inverseSumTerm(src.bValues[k*width + l], a, b, k, l, height, width);
					complex<double> valG = inverseSumTerm(src.gValues[k*width + l], a, b, k, l, height, width);
					complex<double> valR = inverseSumTerm(src.rValues[k*width + l], a, b, k, l, height, width);

					sumB += valB;
					sumG += valG;
					sumR += valR;

				}
			}

			complex<double> newB = sumB / (height*width*1.0);
			complex<double> newG = sumG / (height*width*1.0);
			complex<double> newR = sumR / (height*width*1.0);

			dst.at<Vec3b>(a, b) = Vec3b(newB.real(), newG.real(), newR.real());

		}
	}

}

// shuffle quadrants of the fourier image
void shuffleFourierImage(const FourierImage& src, FourierImage& dst) {

	int width = src.width;
	int height = src.height;

	int halfWidth = width / 2;
	int halfHeight = height / 2;

	dst.width = width;
	dst.height = height;
	dst.bValues.resize(width*height);
	dst.gValues.resize(width*height);
	dst.rValues.resize(width*height);

	// swap upper-left and lower-right
	for (int i = 0; i < halfHeight; i++) {
		for (int j = 0; j < halfWidth; j++) {

			// coordinates to swap values at
			int fromX = i;
			int fromY = j;
			int toX = halfHeight + i;
			int toY = halfWidth + j;

			// B
			dst.bValues[fromX*width + fromY] = src.bValues[toX*width + toY];
			dst.bValues[toX*width + toY] = src.bValues[fromX*width + fromY];

			// G
			dst.gValues[fromX*width + fromY] = src.gValues[toX*width + toY];
			dst.gValues[toX*width + toY] = src.gValues[fromX*width + fromY];

			// R
			dst.rValues[fromX*width + fromY] = src.rValues[toX*width + toY];
			dst.rValues[toX*width + toY] = src.rValues[fromX*width + fromY];

		}
	}

	// swap upper-right and lower-left
	for (int i = 0; i < halfHeight; i++) {
		for (int j = halfWidth; j < width; j++) {

			// coordinates to swap values at
			int fromX = i;
			int fromY = j;
			int toX = halfHeight + i;
			int toY = j - halfWidth;

			// B
			dst.bValues[fromX*width + fromY] = src.bValues[toX*width + toY];
			dst.bValues[toX*width + toY] = src.bValues[fromX*width + fromY];

			// G
			dst.gValues[fromX*width + fromY] = src.gValues[toX*width + toY];
			dst.gValues[toX*width + toY] = src.gValues[fromX*width + fromY];

			// R
			dst.rValues[fromX*width + fromY] = src.rValues[toX*width + toY];
			dst.rValues[toX*width + toY] = src.rValues[fromX*width + fromY];

		}
	}

}

// logarithmic transformation for fourier images
void fourierImageToImage(const FourierImage& src, Mat& dst) {

	int width = src.width;
	int height = src.height;

	dst = Mat(Size(width, height), CV_8UC3);

	double maxMagnitudeB = 0;
	double maxMagnitudeG = 0;
	double maxMagnitudeR = 0;

	for (int k = 0; k < height; k++) {
		for (int l = 0; l < width; l++) {

			if (abs(src.bValues[k*width + l]) > maxMagnitudeB) {
				maxMagnitudeB = abs(src.bValues[k*width + l]);
			}

			if (abs(src.gValues[k*width + l]) > maxMagnitudeG) {
				maxMagnitudeG = abs(src.gValues[k*width + l]);
			}

			if (abs(src.rValues[k*width + l]) > maxMagnitudeR) {
				maxMagnitudeR = abs(src.rValues[k*width + l]);
			}

		}
	}

	double cB = 255.0 / log(1.0 + maxMagnitudeB);
	double cG = 255.0 / log(1.0 + maxMagnitudeG);
	double cR = 255.0 / log(1.0 + maxMagnitudeR);

	for (int k = 0; k < height; k++) {
		for (int l = 0; l < width; l++) {

			double valB = cB * log(1.0 + abs(src.bValues[k*width + l]));
			double valG = cG * log(1.0 + abs(src.gValues[k*width + l]));
			double valR = cR * log(1.0 + abs(src.rValues[k*width + l]));

			Vec3b color(valB, valG, valR);

			dst.at<Vec3b>(k, l) = color;

		}
	}

}

// ideal low pass filter
void idealLowPassFilter(const FourierImage& src, FourierImage& dst, double cuttingFrequency) {

	int width = src.width;
	int height = src.height;

	int originX = height / 2;
	int originY = width / 2;

	dst.width = width;
	dst.height = height;
	dst.bValues.resize(width*height);
	dst.gValues.resize(width*height);
	dst.rValues.resize(width*height);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			double distance = (originX - i)*(originX - i) + (originY - j)*(originY - j);

			if (distance <= cuttingFrequency * cuttingFrequency) {

				dst.bValues[i*width + j] = src.bValues[i*width + j];
				dst.gValues[i*width + j] = src.gValues[i*width + j];
				dst.rValues[i*width + j] = src.rValues[i*width + j];

			}
			else {

				dst.bValues[i*width + j] = complex<double>(0.0, 0.0);
				dst.gValues[i*width + j] = complex<double>(0.0, 0.0);
				dst.rValues[i*width + j] = complex<double>(0.0, 0.0);

			}

		}
	}

}

// ideal high pass filter
void idealHighPassFilter(const FourierImage& src, FourierImage& dst, double cuttingFrequency) {

	int width = src.width;
	int height = src.height;

	int originX = height / 2;
	int originY = width / 2;

	dst.width = width;
	dst.height = height;
	dst.bValues.resize(width*height);
	dst.gValues.resize(width*height);
	dst.rValues.resize(width*height);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			double distance = (originX - i)*(originX - i) + (originY - j)*(originY - j);

			if (distance >= cuttingFrequency * cuttingFrequency) {

				dst.bValues[i*width + j] = src.bValues[i*width + j];
				dst.gValues[i*width + j] = src.gValues[i*width + j];
				dst.rValues[i*width + j] = src.rValues[i*width + j];

			}
			else {

				dst.bValues[i*width + j] = complex<double>(0.0, 0.0);
				dst.gValues[i*width + j] = complex<double>(0.0, 0.0);
				dst.rValues[i*width + j] = complex<double>(0.0, 0.0);

			}

		}
	}

}