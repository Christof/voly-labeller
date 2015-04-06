#include "./label_node.h"
#include <QPainter>

LabelNode::LabelNode(Label label, Gl *gl)
  : label(label), gl(gl)
{
}

LabelNode::~LabelNode()
{
}

void LabelNode::render(const RenderData &renderData)
{
  QPainter painter;
  painter.begin(gl->paintDevice);
  painter.setPen(Qt::blue);
  painter.setFont(QFont("Arial", 16));
  painter.drawText(QRectF(10, 30, 300, 20), Qt::AlignLeft, "Label 1");
  painter.end();
}
