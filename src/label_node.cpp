#include "./label_node.h"
#include <QPainter>
#include <QPoint>
#include "./importer.h"
#include "./mesh.h"

LabelNode::LabelNode(Label label) : label(label)
{
  Importer importer;

  anchorMesh = importer.import("../assets/sphere.dae", 0);
}

LabelNode::~LabelNode()
{
}

void LabelNode::render(Gl *gl, const RenderData &renderData)
{
  anchorMesh->render(gl, renderData.projectionMatrix, renderData.viewMatrix);
  QPainter painter;
  painter.begin(gl->paintDevice);
  painter.setPen(Qt::blue);
  painter.setFont(QFont("Arial", 16));
  painter.drawText(QRectF(10, 30, 300, 20), Qt::AlignLeft, "Label 1");

  Eigen::Vector4f anchorPosition4D;
  anchorPosition4D.head<3>() = label.anchorPosition;
  anchorPosition4D.w() = 1.0f;
  auto anchorPosition2D =
      renderData.projectionMatrix * renderData.viewMatrix * anchorPosition4D;
  QPoint anchorScreen(
      (anchorPosition2D.x() / anchorPosition2D.w() * 0.5f + 0.5f) *
          gl->size.width(),
      (anchorPosition2D.y() / anchorPosition2D.w() * -0.5f + 0.5f) *
          gl->size.height());
  painter.drawArc(QRect(anchorScreen, QSize(4, 4)), 0, 5760);
  painter.drawPoint(anchorScreen);
  painter.end();
}
