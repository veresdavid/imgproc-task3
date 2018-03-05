#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "functions.h"

using namespace cv;
using namespace std;

int main() {

	// load image
	Mat original = imread("D:\\images\\lena_50.png");
	// Mat original = imread("D:\\images\\lena_150.png");
	imshow("Original", original);
	imwrite("D:\\images\\task3\\01-original.png", original);

	// fourier transformation
	FourierImage fourierImage;
	fourierTransformation(original, fourierImage);

	// tmp1
	Mat tmp1;
	fourierImageToImage(fourierImage, tmp1);
	imshow("TMP1", tmp1);
	imwrite("D:\\images\\task3\\02-tmp1.png", tmp1);

	// shuffle
	FourierImage shuffled;
	shuffleFourierImage(fourierImage, shuffled);

	// display fourier image
	Mat frequencyImage;
	fourierImageToImage(shuffled, frequencyImage);
	imshow("Frequency image", frequencyImage);
	imwrite("D:\\images\\task3\\03-frequency.png", frequencyImage);

	// ideal low pass filter
	FourierImage filteredFourierImage;
	idealLowPassFilter(shuffled, filteredFourierImage, 12.5);
	// idealLowPassFilter(shuffled, filteredFourierImage, 37.5);

	// ideal high pass filter
	// FourierImage filteredFourierImage;
	// idealHighPassFilter(shuffled, filteredFourierImage, 10);

	// display new fourier image
	Mat filteredFrequencyImage;
	fourierImageToImage(filteredFourierImage, filteredFrequencyImage);
	imshow("Filtered frequency image", filteredFrequencyImage);
	imwrite("D:\\images\\task3\\04-filtered_frequency.png", filteredFrequencyImage);

	// shuffle back
	FourierImage newFourierImage;
	shuffleFourierImage(filteredFourierImage, newFourierImage);

	// tmp2
	Mat tmp2;
	fourierImageToImage(newFourierImage, tmp2);
	imshow("TMP2", tmp2);
	imwrite("D:\\images\\task3\\05-tmp2.png", tmp2);

	// invert fourier transformation
	Mat newImage;
	inverseFourierTransformation(newFourierImage, newImage);
	imshow("New image", newImage);
	imwrite("D:\\images\\task3\\06-result.png", newImage);

	// wait
	waitKey();

	return 0;
}