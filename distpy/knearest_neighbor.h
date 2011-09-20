#ifndef KNEAREST_NEIGHBOR_H
#define KNEAREST_NEIGHBOR_H
int knnl2sqr(double *test_point, double *train_points, int *neighbor_indeces, double *neighbor_dists, int num_train_points, int num_dims, int num_neighbors);
void nnsl2sqr(double *test_points, double *train_points, double *neighbor_dist_indeces, int num_test_points, int num_train_points, int num_dims);
#endif
