#include "./cost_function_calculator.h"
#include <thrust/transform_reduce.h>
#include <limits>

struct EvalResult
{
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
  {
  }
  __device__ EvalResult operator()(const int &index) const
  {
    EvalResult result;
    result.cost = 0;

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

CostFunctionCalculator::CostFunctionCalculator()
{
}

void CostFunctionCalculator::calculateForLabel()
{
  CostEvaluator costEvaluator(width, height);

  MinimumCostOperator<EvalResult> minimumCostOperator;
  EvalResult initialCost;
  initialCost.x = -1;
  initialCost.y = -1;
  initialCost.cost = std::numeric_limits<float>::max();

  EvalResult cost = thrust::transform_reduce(
      thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(0) + width * height, costEvaluator,
      initialCost, minimumCostOperator);
}

