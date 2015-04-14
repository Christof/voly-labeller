#include "./label_node.h"
#include <QPainter>
#include <QPoint>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "./importer.h"
#include "./mesh.h"

LabelNode::LabelNode(Label label) : label(label)
{
  Importer importer;

  anchorMesh = importer.import("../assets/anchor.dae", 0);
}

LabelNode::~LabelNode()
{
}

void LabelNode::render(Gl *gl, RenderData renderData)
{
  Eigen::Affine3f transform(Eigen::Translation3f(label.anchorPosition) *
                            Eigen::Scaling(0.005f));
  renderData.modelMatrix = transform.matrix();
  anchorMesh->render(gl, renderData);
  QPainter painter;
  painter.begin(gl->paintDevice);
  painter.setPen(Qt::blue);
  painter.setFont(QFont("Arial", 16));
  painter.drawText(QRectF(10, 30, 300, 20), Qt::AlignLeft, "Label 1");

  painter.end();
}
