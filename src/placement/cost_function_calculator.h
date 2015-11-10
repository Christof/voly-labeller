#ifndef SRC_PLACEMENT_COST_FUNCTION_CALCULATOR_H_

#define SRC_PLACEMENT_COST_FUNCTION_CALCULATOR_H_

#include <thrust/device_vector.h>
#include <tuple>

/**
 * \brief
 *
 *
 */
class CostFunctionCalculator
{
 public:
  CostFunctionCalculator() = default;

  void resize(int width, int height);
  void setTextureSize(int width, int height);

  void calculateCosts(const thrust::device_vector<float> &distances);
  std::tuple<float, float> calculateForLabel(
      const thrust::device_vector<float> &occupancySummedAreaTable, int labelId,
      float anchorX, float anchorY, int sizeX, int sizeY);

 private:
  int width;
  int height;

  int textureWidth;
  int textureHeight;
};

#endif  // SRC_PLACEMENT_COST_FUNCTION_CALCULATOR_H_
