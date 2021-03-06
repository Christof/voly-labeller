#include "./obb_node.h"
#include <Eigen/Core>
#include <vector>

ObbNode::ObbNode(const Math::Obb &obb)
{
  wireframe =
      std::make_shared<Graphics::Connector>(std::vector<Eigen::Vector3f>{
        obb.corners[0], obb.corners[1], obb.corners[1], obb.corners[2],
        obb.corners[2], obb.corners[3], obb.corners[3], obb.corners[0],
        obb.corners[4], obb.corners[5], obb.corners[5], obb.corners[6],
        obb.corners[6], obb.corners[7], obb.corners[7], obb.corners[4],
        obb.corners[0], obb.corners[4], obb.corners[1], obb.corners[5],
        obb.corners[2], obb.corners[6], obb.corners[3], obb.corners[7]
      });
  wireframe->color = Eigen::Vector4f(0.1f, 0.5f, 0.5f, 1);
  wireframe->lineWidth = 1.0f;
}

void ObbNode::render(Graphics::Gl *gl,
                     std::shared_ptr<Graphics::Managers> managers,
                     RenderData renderData)
{
  wireframe->render(gl, managers, renderData);
}

