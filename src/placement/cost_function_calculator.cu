#if _WIN32
#pragma warning(disable : 4267 4244)
#endif

#include "./cost_function_calculator.h"
#include <thrust/transform_reduce.h>
#include <limits>
#include <cfloat>
#include <tuple>
#include "./placement.h"

struct EvalResult
{
  __host__ __device__ EvalResult()
  {
  }

  __host__ __device__ EvalResult(int x, int y, float cost)
    : x(x), y(y), cost(cost)
  {
  }

  int x, y;

  float cost;
};

__host__ __device__ bool operator<(const EvalResult &a, const EvalResult &b)
{
  return a.cost < b.cost;
}

struct CostEvaluator : public thrust::unary_function<int, EvalResult>
{
  __host__ __device__ CostEvaluator(int width, int height,
                                    Placement::CostFunctionWeights weights,
                                    unsigned char labelShadowValue,
                                    unsigned char connectorShadowValue,
                                    unsigned char anchorConstraintValue)
    : width(width), height(height), weights(weights),
      labelShadowValue(labelShadowValue),
      connectorShadowValue(connectorShadowValue),
      anchorConstraintValue(anchorConstraintValue)
  {
  }

  int width;
  int height;
  Placement::CostFunctionWeights weights;
  const unsigned char labelShadowValue;
  const unsigned char connectorShadowValue;
  const unsigned char anchorConstraintValue;

  int halfLabelWidth;
  int halfLabelHeight;

  float anchorX;
  float anchorY;

  int oldPositionX;
  int oldPositionY;

  cudaTextureObject_t constraints;
  const float *integralCosts;

  __device__ float lineLength(int x, int y) const
  {
    float diffX = x - anchorX;
    float diffY = y - anchorY;

    return sqrt(diffX * diffX + diffY * diffY);
  }

  __device__ float distanceToOldPosition(int x, int y) const
  {
    float diffX = x - oldPositionX;
    float diffY = y - oldPositionY;

    return sqrt(diffX * diffX + diffY * diffY);
  }

  __device__ float favorHorizontalOrVerticalLines(int x, int y) const
  {
    float diffX = x - anchorX;
    float diffY = y - anchorY;

    float length = sqrt(diffX * diffX + diffY * diffY);
    diffX = diffX / length;
    diffY = diffY / length;

    return fabs(diffX) + fabs(diffY);
  }

  __device__ float integralCostsForLabelArea(int x, int y) const
  {
    int startX = x - halfLabelWidth - 1;
    int startY = y - halfLabelHeight - 1;
    int endX = x + halfLabelWidth;
    int endY = y + halfLabelHeight;

    float lowerRight = integralCosts[endY * width + endX];
    float lowerLeft = integralCosts[endY * width + startX];
    float upperLeft = integralCosts[startY * width + startX];
    float upperRight = integralCosts[startY * width + endX];
    float sum = lowerRight + upperLeft - lowerLeft - upperRight;

    // this can be the case through inaccuracies
    if (sum < 0.0f)
      return 0.0f;

    return sum / ((endX - startX) * (endY - startY));
  }

  __device__ EvalResult operator()(const int &index) const
  {
    int x = index % width;
    int y = index / width;
    const int border = 1;

    if (x < halfLabelWidth + border || x > width - halfLabelWidth - border ||
        y < halfLabelHeight + border || y > height - halfLabelHeight - border)
      return EvalResult(x, y, FLT_MAX);

    unsigned char constraintValue =
        tex2D<unsigned char>(constraints, x + 0.5f, y + 0.5f);

    float distanceToAnchor = lineLength(x, y);

    float labelShadow = constraintValue & labelShadowValue ? 1.0f : 0.0f;
    float connectorShadow = constraintValue & connectorShadowValue ?
      1.0f : 0.0f;
    float anchorConstraint = constraintValue & anchorConstraintValue ?
      1.0f : 0.0f;

    float cost = weights.labelShadowConstraint * labelShadow +
                 weights.connectorShadowConstraint * connectorShadow +
                 weights.anchorConstraint * anchorConstraint +
                 weights.integralCosts * integralCostsForLabelArea(x, y) +
                 weights.distanceToAnchor * distanceToAnchor +
                 weights.distanceToOldPosition * distanceToOldPosition(x, y) +
                 weights.favorHorizontalOrVerticalLines *
                     favorHorizontalOrVerticalLines(x, y);

    EvalResult result(x, y, cost);
    return result;
  }
};

template <typename T>
struct MinimumCostOperator : public thrust::binary_function<T, T, T>  // NOLINT
{
  __host__ __device__ T operator()(const T &x, const T &y) const
  {
    return x < y ? x : y;
  }
};

namespace Placement
{

CostFunctionCalculator::CostFunctionCalculator(
    std::shared_ptr<CudaArrayProvider> constraintImage)
  : constraintImage(constraintImage)
{
}

void CostFunctionCalculator::resize(int width, int height)
{
  this->width = width;
  this->height = height;
}

void CostFunctionCalculator::setTextureSize(int width, int height)
{
  textureWidth = width;
  textureHeight = height;
}

CostFunctionResult CostFunctionCalculator::calculateForLabel(
    const thrust::device_vector<float> &integralCosts, int labelId,
    float anchorX, float anchorY, int labelWidthInPixel, int labelHeightInPixel,
    bool ignoreOldPosition, int oldPositionX, int oldPositionY)
{
  createTextureObject();

  assert(textureWidth * textureHeight == integralCosts.size());

  float widthFactor = static_cast<float>(textureWidth) / width;
  float heightFactor = static_cast<float>(textureHeight) / height;

  CostFunctionWeights usedWeights = weights;

  if (ignoreOldPosition)
    usedWeights.distanceToOldPosition = 0.0f;
  else
    usedWeights.distanceToAnchor = 0.04f;

  CostEvaluator costEvaluator(
      textureWidth, textureHeight, usedWeights, Placement::labelShadowValue,
      Placement::connectorShadowValue, Placement::anchorConstraintValue);
  costEvaluator.anchorX = anchorX * widthFactor;
  costEvaluator.anchorY = anchorY * heightFactor;
  costEvaluator.integralCosts = thrust::raw_pointer_cast(integralCosts.data());
  costEvaluator.constraints = constraints;
  costEvaluator.halfLabelWidth =
      static_cast<int>(labelWidthInPixel * 0.5f * widthFactor);
  costEvaluator.halfLabelHeight =
      static_cast<int>(labelHeightInPixel * 0.5f * heightFactor);
  costEvaluator.oldPositionX = oldPositionX * widthFactor;
  costEvaluator.oldPositionY = oldPositionY * widthFactor;

  MinimumCostOperator<EvalResult> minimumCostOperator;
  EvalResult initialCost;
  initialCost.x = -1;
  initialCost.y = -1;
  initialCost.cost = std::numeric_limits<float>::max();

  EvalResult cost = thrust::transform_reduce(
      thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(0) + textureWidth * textureHeight,
      costEvaluator, initialCost, minimumCostOperator);

  // std::cout << cost.x << "/" << cost.y << ": " << cost.cost << std::endl;

  cudaDestroyTextureObject(constraints);

  if (isnan(cost.cost))
    return CostFunctionResult{ Eigen::Vector2i(textureWidth - 1, 0),
                               cost.cost };

  if (cost.cost > 1e32)
    return CostFunctionResult{ Eigen::Vector2i(0, 0), cost.cost };

  return CostFunctionResult{ Eigen::Vector2i(cost.x, cost.y), cost.cost };
}

void CostFunctionCalculator::createTextureObject()
{
  constraintImage->map();

  auto resDesc = constraintImage->getResourceDesc();
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaCreateTextureObject(&constraints, &resDesc, &texDesc, NULL);

  constraintImage->unmap();
}

}  // namespace Placement
