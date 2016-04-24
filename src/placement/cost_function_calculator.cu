#include "./cost_function_calculator.h"
#include <thrust/transform_reduce.h>
#include <limits>
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

  bool operator<(const EvalResult &other)
  {
    return thrust::min<float>(this->cost, other.cost);
  }
};

__host__ __device__ bool operator<(const EvalResult &a, const EvalResult &b)
{
  return (a.cost < b.cost);
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

  cudaTextureObject_t constraints;
  const float *occupancy;

  __device__ float lineLength(int x, int y) const
  {
    float diffX = x - anchorX;
    float diffY = y - anchorY;

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

  __device__ float occupancyForLabelArea(int x, int y) const
  {
    int startX = max(x - halfLabelWidth - 1, 0);
    int startY = max(y - halfLabelWidth - 1, 0);
    int endX = min(x + halfLabelWidth, width - 1);   // NOLINT
    int endY = min(y + halfLabelWidth, height - 1);  // NOLINT

    float lowerRight = occupancy[endY * width + endX];
    float lowerLeft = occupancy[endY * width + startX];
    float upperLeft = occupancy[startY * width + startX];
    float upperRight = occupancy[startY * width + endX];
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

    unsigned char constraintValue =
        tex2D<unsigned char>(constraints, x + 0.5f, y + 0.5f);

    float distanceToAnchor = lineLength(x, y);

    unsigned char labelShadow = constraintValue & labelShadowValue;
    unsigned char connectorShadow = constraintValue & connectorShadowValue;
    unsigned char anchorConstraint = constraintValue & anchorConstraintValue;

    float cost = weights.labelShadowConstraint * labelShadow +
                 weights.connectorShadowConstraint * connectorShadow +
                 weights.anchorConstraint * anchorConstraint +
                 weights.occupancy * occupancyForLabelArea(x, y) +
                 weights.distanceToAnchor * distanceToAnchor +
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
    T result;

    result = x < y ? x : y;
    return result;
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

std::tuple<float, float> CostFunctionCalculator::calculateForLabel(
    const thrust::device_vector<float> &integralCosts, int labelId,
    float anchorX, float anchorY, int labelWidthInPixel, int labelHeightInPixel)
{
  createTextureObject();

  assert(textureWidth * textureHeight == integralCosts.size());

  float widthFactor = static_cast<float>(textureWidth) / width;
  float heightFactor = static_cast<float>(textureHeight) / height;

  CostEvaluator costEvaluator(textureWidth, textureHeight, weights,
                              Placement::labelShadowValue,
                              Placement::connectorShadowValue,
                              Placement::anchorConstraintValue);
  costEvaluator.anchorX = anchorX * widthFactor;
  costEvaluator.anchorY = anchorY * heightFactor;
  costEvaluator.occupancy =
      thrust::raw_pointer_cast(integralCosts.data());
  costEvaluator.constraints = constraints;
  costEvaluator.halfLabelWidth = labelWidthInPixel * 0.5f * widthFactor;
  costEvaluator.halfLabelHeight = labelHeightInPixel * 0.5f * heightFactor;

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

  return std::make_tuple(cost.x, cost.y);
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
