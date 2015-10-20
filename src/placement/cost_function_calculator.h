#ifndef SRC_PLACEMENT_COST_FUNCTION_CALCULATOR_H_

#define SRC_PLACEMENT_COST_FUNCTION_CALCULATOR_H_

#include <thrust/device_vector.h>

/**
 * \brief
 *
 *
 */
class CostFunctionCalculator
{
 public:
  CostFunctionCalculator(int width, int height);

  void calculateCosts(const thrust::device_vector<float> &distances);
  void calculateForLabel(const thrust::device_vector<float> &distances,
                         int labelId, float anchorX, float anchorY);

 private:
  int width;
  int height;
};

#endif  // SRC_PLACEMENT_COST_FUNCTION_CALCULATOR_H_
