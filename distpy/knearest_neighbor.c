#include <math.h>
#include <stdlib.h>

int place_dist_in_results(const int index, const double dist, int *neighbor_indeces, double *neighbor_dists, const int num_neighbors, const int max_valid_ind) {
  int i, j;
  /* Quick check, if this dist is greater than our worst neighbor, then return */
  if (dist >= neighbor_dists[num_neighbors - 1])
    return max_valid_ind;
  /* Find the first spot where we can put this neighbor, move everything else down */
  for (i = 0; i < num_neighbors; ++i) {
    if (dist < neighbor_dists[i]) {
      int start_ind = max_valid_ind;
      if (start_ind == num_neighbors - 1)
	start_ind = num_neighbors - 2;
      for (j = start_ind; j >= i; --j) { /* Move down */
	neighbor_indeces[j+1] = neighbor_indeces[j];
	neighbor_dists[j+1] = neighbor_dists[j];
      }
      neighbor_indeces[i] = index;
      neighbor_dists[i] = dist;
      return max_valid_ind == num_neighbors - 1 ? max_valid_ind : max_valid_ind + 1;
    }
  }
  return max_valid_ind; /* Shouldn't get here because of the initial check */
}

static inline void clear_arrays(int *neighbor_indeces, double *neighbor_dists, int num_neighbors) {
  int i;
  for (i = 0; i < num_neighbors; ++i) {
    neighbor_indeces[i] = -1;
    neighbor_dists[i] = HUGE_VAL;
  }
}

void knnl1(double *test_point, double *train_points, int *neighbor_indeces, double *neighbor_dists, int num_train_points, int num_dims, int num_neighbors) {
  int i, j, max_valid_ind = -1;
  clear_arrays(neighbor_indeces, neighbor_dists, num_neighbors);
  for (i = 0; i < num_train_points; ++i, train_points += num_dims) {
    double cur_dist = 0.0;
    for (j = 0; j < num_dims; ++j)
      cur_dist += fabs(test_point[j] - train_points[j]);
    max_valid_ind = place_dist_in_results(i, cur_dist, neighbor_indeces, neighbor_dists, num_neighbors, max_valid_ind);
  }
}

int knnl2sqr(double *test_point, double *train_points, int *neighbor_indeces, double *neighbor_dists, int num_train_points, int num_dims, int num_neighbors) {
  int i, j, max_valid_ind = -1;
  double *temp_dist = malloc(sizeof *test_point * num_dims);
  clear_arrays(neighbor_indeces, neighbor_dists, num_neighbors);
  for (i = 0; i < num_train_points; ++i, train_points += num_dims) {
    double cur_dist = 0.0;
    for (j = 0; j < num_dims; ++j)
      temp_dist[j] = (test_point[j] - train_points[j]) * (test_point[j] - train_points[j]);
    for (j = 0; j < num_dims; ++j)
      cur_dist += temp_dist[j];
    max_valid_ind = place_dist_in_results(i, cur_dist, neighbor_indeces, neighbor_dists, num_neighbors, max_valid_ind);
  }
  free(temp_dist);
  return max_valid_ind;
}

void nnsl2sqr(double *test_points, double *train_points, double *neighbor_dist_indeces, int num_test_points, int num_train_points, int num_dims) {
  int i, j, k;
  double *train_points_orig = train_points;
  for (i = 0; i < num_test_points; ++i, test_points += num_dims, neighbor_dist_indeces += 2) {
      double cur_dist = HUGE_VAL;
      int cur_ind = 0;
      for (j = 0, train_points = train_points_orig; j < num_train_points; ++j, train_points += num_dims) {
          double temp_dist = 0.;
          for (k = 0; k < num_dims; ++k)
              temp_dist += (test_points[k] - train_points[k]) * (test_points[k] - train_points[k]);
          if (temp_dist < cur_dist) {
              cur_dist = temp_dist;
              cur_ind = j;
          }
      }
      neighbor_dist_indeces[0] = cur_dist;
      neighbor_dist_indeces[1] = cur_ind;
  }
}
