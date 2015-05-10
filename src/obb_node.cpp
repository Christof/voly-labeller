#include "./obb_node.h"
#include <Eigen/Core>
#include <vector>
#include "./connector.h"

ObbNode::ObbNode(std::shared_ptr<Math::Obb> obb)
{
  wireframe = std::make_shared<Connector>(std::vector<Eigen::Vector3f>{
    obb->corners[0], obb->corners[1], obb->corners[1], obb->corners[2],
    obb->corners[2], obb->corners[3], obb->corners[3], obb->corners[0],
    obb->corners[4], obb->corners[5], obb->corners[5], obb->corners[6],
    obb->corners[6], obb->corners[7], obb->corners[7], obb->corners[4],
    obb->corners[0], obb->corners[4], obb->corners[1], obb->corners[5],
    obb->corners[2], obb->corners[6], obb->corners[3], obb->corners[7]
  });
  wireframe->color = Eigen::Vector4f(0.1f, 0.5f, 0.5f, 1);
  wireframe->lineWidth = 1.0f;
}

void ObbNode::render(Gl *gl, RenderData renderData)
{
  wireframe->render(gl, renderData);
}

