#ifndef __FACE_DETECTION__
#define __FACE_DETECTION__

typedef struct
{
	int left_top_x, left_top_y, right_bottom_x, right_bottom_y;
}face_box_t;

/*
	face detection init, create session and load model

	return : 
		0  -- success
		-1 -- failed
*/
int face_detection_init(const char *model_path);

/*
	detection face from the image

	return:
		0  : success
		-1 : have no face
		return_box : location of the rectangle
		prob : probability
*/
int face_detection(unsigned char *data, int height, int width, face_box_t *return_box, float *prob);

#endif