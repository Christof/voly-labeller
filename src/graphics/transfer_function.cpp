#include "./transfer_function.h"
#include <QGradient>
#include <QString>
#include <QDomDocument>
#include <QFile>
#include <QLoggingCategory>
#include <qtgradienteditor/qtgradientutils.h>
#include <stdexcept>

namespace Graphics
{

// QLoggingCategory tfChan("Graphics.TransferFunction");

TransferFunction::TransferFunction(std::string path)
{
  QDomDocument doc;
  QFile file(path.c_str());
  if (!file.open(QIODevice::ReadOnly))
  {
    throw std::runtime_error("Could not open file" + path);
  }
  if (!doc.setContent(&file))
  {
    file.close();
    throw std::runtime_error("Could not read content from file" + path);
  }
  file.close();
  QGradient gradient = QtGradientUtils::loadGradient(doc.firstChildElement());
}

TransferFunction::~TransferFunction
{
}

}  // namespace Graphics
