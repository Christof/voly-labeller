#ifndef SRC_PLACEMENT_PLACEMENT_H_

#define SRC_PLACEMENT_PLACEMENT_H_

/**
 * \brief Contains classes for label placement using a global minimization of a
 * cost function
 *
 */
namespace Placement
{

constexpr unsigned char labelShadowValue = 1 << (7 - 0);
constexpr unsigned char connectorShadowValue = 1 << (7 - 1);
constexpr unsigned char anchorConstraintValue = 1 << (7 - 2);

}  // namespace Placement

#endif  // SRC_PLACEMENT_PLACEMENT_H_
