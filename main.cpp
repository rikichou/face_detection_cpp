#include <iostream>
#include <time.h>

#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/default/integral_types.h>
#include <opencv2/opencv.hpp>

#include "face_detecion.h"

using namespace std;
using namespace tensorflow;
using namespace cv;

int main()
{
	/* init */
	face_detection_init("/home/tensortec/riki/workspace/pro/object_detection/ssd/face_tiny/predict_model/frozen_inference_graph.pb");

	/* predict */
	const char *img_path = "/home/tensortec/riki/workspace/pro/object_detection/ssd/multi_obj/images/call/2018-07-19  20.22.37_939.jpg";

	Mat img_mat = imread(img_path,0);
	if (img_mat.empty())
	{
		cout << "Can not open the image " << img_path << endl;
		return -1;
	}

	float prob=0;
	face_box_t box;
	cout << img_mat.size().height << " " << img_mat.size().width << endl;
	for (int i = 0; i < 100; i ++)
	{
		cout << ((unsigned char *)img_mat.data)[i] << " ";
	}
	cout << endl;
	int ret = face_detection(img_mat.data, img_mat.size().height, img_mat.size().width, &box, &prob);

	if (ret != 0)
	{
		cout << "Can not detect any face~" << endl;
		return -1;
	}
	else
	{
		cout << "Detect face, and probability is " << prob << endl;
	}
	
	return 0;
}


