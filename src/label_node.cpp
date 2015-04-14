#include "./label_node.h"
#include <QPainter>
#include <QImage>
#include <QPoint>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "./importer.h"
#include "./mesh.h"
#include "./quad.h"

LabelNode::LabelNode(Label label) : label(label)
{
  Importer importer;

  anchorMesh = importer.import("../assets/anchor.dae", 0);
  quad = std::make_shared<Quad>();
}

LabelNode::~LabelNode()
{
}

void LabelNode::render(Gl *gl, RenderData renderData)
{
  if (!loadedText)
    renderLabelTextToTexture(gl);

  Eigen::Affine3f transform(Eigen::Translation3f(label.anchorPosition) *
                            Eigen::Scaling(0.005f));
  renderData.modelMatrix = transform.matrix();
  anchorMesh->render(gl, renderData);

  renderData.modelMatrix = Eigen::Matrix4f();
  quad->render(gl, renderData);
}

void LabelNode::renderLabelTextToTexture(Gl *gl)
{
  QImage *image = new QImage(512, 128, QImage::Format_ARGB32);
  image->fill(Qt::GlobalColor::transparent);

  QPainter painter;
  painter.begin(image);
  painter.setPen(Qt::blue);
  painter.setFont(QFont("Arial", 72));
  painter.drawText(QRectF(0, 0, 512, 128), Qt::AlignCenter, label.text.c_str());
  painter.end();

  image->save("test.png");

  delete image;
  loadedText = true;
}
