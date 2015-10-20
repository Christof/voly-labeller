#ifndef SRC_PLACEMENT_COST_FUNCTION_CALCULATOR_H_

#define SRC_PLACEMENT_COST_FUNCTION_CALCULATOR_H_

/**
 * \brief
 *
 *
 */
class CostFunctionCalculator
{
 public:
  CostFunctionCalculator();

  void calculateForLabel();

 private:
  int width;
  int height;
};

#endif  // SRC_PLACEMENT_COST_FUNCTION_CALCULATOR_H_
