#include "./cost_function_calculator.h"
#include <thrust/transform_reduce.h>
#include <limits>

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
  __host__ __device__ CostEvaluator(int width, int height)
    : width(width), height(height)
  {
  }

  int width;
  int height;

  float anchorX;
  float anchorY;

  const float *distances;

  __device__ EvalResult operator()(const int &index) const
  {
    int x = index % width;
    int y = index / width;

    float xDistance = x - anchorX;
    float yDistance = y - anchorY;
    float distanceToAnchor = xDistance * xDistance + yDistance * yDistance;

    float cost = distanceToAnchor + 10.0f * distances[index];
    EvalResult result(x, y, cost);

    return result;
  }
};

template <typename T>
struct MinimumCostOperator : public thrust::binary_function<T, T, T>
{
  __host__ __device__ T operator()(const T &x, const T &y) const
  {
    T result;

    result = x < y ? x : y;
    return result;
  }
};

CostFunctionCalculator::CostFunctionCalculator(int width, int height)
  : width(width), height(height)
{
}

void CostFunctionCalculator::calculateCosts(
    const thrust::device_vector<float> &distances)
{
  calculateForLabel(distances, 0, 500, 500);
}

std::tuple<float, float> CostFunctionCalculator::calculateForLabel(
    const thrust::device_vector<float> &distances, int labelId, float anchorX,
    float anchorY)
{
  CostEvaluator costEvaluator(width, height);
  costEvaluator.anchorX = anchorX;
  costEvaluator.anchorY = anchorY;
  costEvaluator.distances = thrust::raw_pointer_cast(distances.data());

  MinimumCostOperator<EvalResult> minimumCostOperator;
  EvalResult initialCost;
  initialCost.x = -1;
  initialCost.y = -1;
  initialCost.cost = std::numeric_limits<float>::max();

  EvalResult cost = thrust::transform_reduce(
      thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(0) + width * height, costEvaluator,
      initialCost, minimumCostOperator);

  std::cout << cost.x << "/" << cost.y << ": " << cost.cost << std::endl;
  return std::make_tuple(cost.x, cost.y);
}

