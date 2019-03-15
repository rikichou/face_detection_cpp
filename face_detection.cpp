
#include <iostream>
#include <time.h>

#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/default/integral_types.h>
#include <opencv2/opencv.hpp>

#include "face_detection.h"

using namespace std;
using namespace tensorflow;
using namespace cv;

/* global session */
Session *g_session = NULL;

/* global graph */
GraphDef g_graph_def;

/*
	face detection init, create session and load model

	return : 
		0  -- success
		-1 -- failed
*/
int face_detection_init(const char *model_path)
{
	/* create session */
	Status status = NewSession(SessionOptions(), &g_session);
	if (!status.ok())
	{
		cout << status.ToString() << "\n";
	    return -1;
	}

	/* load model */
	status = ReadBinaryProto(Env::Default(), model_path, &g_graph_def);
	if (!status.ok())
	{
		cout << status.ToString() << "\n";
		return -1;
	}

	status = g_session->Create(g_graph_def);
	if (!status.ok())
	{
		cout << status.ToString() << "\n";
	    return -1;
	}

	return 0;
}

/*
	detection face from the image

	return 
*/
int face_detection(unsigned char *data, int height, int width, face_box_t *return_box, float *prob)
{
	/* data into cv MAT */
	cv::Mat mat(height, width, CV_8UC1, data);
	cv::Mat img_mat;

	cvtColor(mat, img_mat, COLOR_GRAY2RGB);

	// alloc tensor
	Tensor input_tensor(DT_UINT8, TensorShape({1, img_mat.size().height, img_mat.size().width, 3}));

	// fill tensor
	uint8 *p = input_tensor.flat<uint8>().data();
	cv::Mat tempMat(img_mat.size().height, img_mat.size().width, CV_8UC3, p);
	img_mat.convertTo(tempMat,CV_8UC3);

	// get output
	vector<Tensor> outputs;

	Status status = g_session->Run({{"image_tensor:0", input_tensor}}, {"num_detections:0", "detection_boxes:0", "detection_scores:0"}, {}, &outputs);
	if (!status.ok())
	{
		cout << __LINE__ << status.ToString() << "\n";
		return -1;
	}

#if 0
	clock_t start, end;
	start = clock();
	for (int i = 0; i < 100; i ++)
	{
		status = session->Run({{"image_tensor:0", input_tensor}}, {"num_detections:0", "detection_boxes:0", "detection_scores:0"}, {}, &outputs);		
	}
	end = clock();
	cout << "100 images and " << (float)(end-start)*1000/CLOCKS_PER_SEC;
#endif

	Tensor score_t = outputs[2], box_t = outputs[1];

	auto score_map = score_t.tensor<float, 2>();
	auto box_map = box_t.tensor<float, 3>();

	if (score_map(0,0) < 0.6)
	{
		cout << "There is no Face!" << endl;
		return -1;
	}

	if (prob)
	{
		*prob = score_map(0,0);
	}

	return_box->left_top_x = (int)box_map(0,0,1)*width;
	return_box->left_top_y = (int)box_map(0,0,0)*height;
	return_box->right_bottom_x = (int)box_map(0,0,3)*width;
	return_box->right_bottom_y = (int)box_map(0,0,2)*height;

	cout << box_map(0,0,0) << box_map(0,0,1) << box_map(0,0,2) << box_map(0,0,3) << endl;

	#if 0
	Rect cv_rect(box_map(0,0,1)*width, box_map(0,0,0)*height,
		(box_map(0,0,3) - box_map(0,0,1))*width, (box_map(0,0,2) - box_map(0,0,0))*height);
	cv::rectangle(img_mat, cv_rect,Scalar(0,0,255),1,1,0);
	imwrite("test.jpg", img_mat);
	#endif

	return 0;
}

