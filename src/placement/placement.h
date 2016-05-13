#ifndef SRC_PLACEMENT_PLACEMENT_H_

#define SRC_PLACEMENT_PLACEMENT_H_

/**
 * \brief Contains classes for label placement using a global minimization of a
 * cost function
 *
 */
namespace Placement
{

const unsigned char labelShadowValue = 1 << (7 - 0);
const unsigned char connectorShadowValue = 1 << (7 - 1);
const unsigned char anchorConstraintValue = 1 << (7 - 2);

}  // namespace Placement

#endif  // SRC_PLACEMENT_PLACEMENT_H_
