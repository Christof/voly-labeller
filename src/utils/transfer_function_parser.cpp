#include "./transfer_function_parser.h"
#include <QFile>
#include <QGradient>
#include <QGradientStops>
#include <QXmlSimpleReader>
#include <QXmlInputSource>
#include <QMetaEnum>
#include <QException>
#include <iostream>

class QGradientContentHandler : public QXmlDefaultHandler
{
 private:
  QGradient *instance;

  QGradientStops stops;
  double stopPosition;
  QColor stopColor;

 public:
  QGradientContentHandler() : QXmlDefaultHandler()
  {
  }

  ~QGradientContentHandler()
  {
  }

  QGradient *getQGradientInstance()
  {
    return instance;
  }

  bool fatalError(const QXmlParseException &exception)
  {
    std::cout << "Fatal error on line" << exception.message().toStdString()
              << std::endl;

    return false;
  }

  QGradient::Spread parseSpread(QString spreadString)
  {
    if (spreadString.compare("PadSpread") == 0)
      return QGradient::Spread::PadSpread;
    if (spreadString.compare("ReflectSpread") == 0)
      return QGradient::Spread::ReflectSpread;
    if (spreadString.compare("RepeatSpread") == 0)
      return QGradient::Spread::RepeatSpread;
    else
      throw new QString("Spread value not supported!");
  }

  QGradient::CoordinateMode parseCoordinateMode(QString coordinateModeString)
  {
    if (coordinateModeString.compare("LogicalMode") == 0)
      return QGradient::CoordinateMode::LogicalMode;
    else if (coordinateModeString.compare("StretchToDeviceMode") == 0)
      return QGradient::CoordinateMode::StretchToDeviceMode;
    else if (coordinateModeString.compare("ObjectBoundingMode") == 0)
      return QGradient::CoordinateMode::ObjectBoundingMode;
    else
      throw new QString("CoordinateMode value not supported!");
  }

  void parseGradientAttributes(const QXmlAttributes &atts)
  {
    if (atts.value("type").compare(("LinearGradient")) == 0)
    {
      auto startX = atts.value("startX").toDouble();
      auto startY = atts.value("startY").toDouble();
      auto endX = atts.value("endX").toDouble();
      auto endY = atts.value("endY").toDouble();

      instance = new QLinearGradient(startX, startY, endX, endY);
    }
    else if (atts.value("type").compare("RadialGradient") == 0)
    {
      auto centerX = atts.value("centerX").toDouble();
      auto centerY = atts.value("centerY").toDouble();
      auto radius = atts.value("radius").toDouble();
      auto fX = atts.value("fX").toDouble();
      auto fY = atts.value("fY").toDouble();

      instance = new QRadialGradient(centerX, centerY, radius, fX, fY);
    }
    else if (atts.value("type").compare("ConicalGradient") == 0)
    {
      auto centerX = atts.value("centerX").toDouble();
      auto centerY = atts.value("centerY").toDouble();
      auto startAngle = atts.value("startAngle").toDouble();

      instance = new QConicalGradient(centerX, centerY, startAngle);
    }
    else
      throw new QString("gradient type not supported!");

    instance->setSpread(parseSpread(atts.value("spread")));
    instance->setCoordinateMode(
        parseCoordinateMode(atts.value("coordinateMode")));

    // reset stops here
    stops.clear();
  }

  void parseStopData(const QXmlAttributes &atts)
  {
    stopPosition = atts.value("position").toDouble();
  }

  void parseColorData(const QXmlAttributes &atts)
  {
    stopColor.setRed(atts.value("r").toInt());
    stopColor.setGreen(atts.value("g").toInt());
    stopColor.setBlue(atts.value("b").toInt());
    stopColor.setAlpha(atts.value("a").toInt());
  }

  bool startElement(const QString &namespaceURI, const QString &localName,
                    const QString &qName, const QXmlAttributes &atts)
  {
    std::cout << "Read Start Tag : " << localName.toStdString() << std::endl;
    if (localName.compare("gradientData") == 0)
      parseGradientAttributes(atts);
    if (localName.compare("stopData") == 0)
      parseStopData(atts);
    if (localName.compare("colorData") == 0)
      parseColorData(atts);

    std::cout << "------------------------" << std::endl;
    return true;
  }

  bool endElement(const QString &namespaceURI, const QString &localName,
                  const QString &qName)
  {
    std::cout << "Read End Element Tag : " << localName.toStdString()
              << std::endl;
    if (localName.compare("gradientData") == 0)
    {
      std::cout << "Setting Stops with size: " << stops.size() << std::endl;
      instance->setStops(stops);
    }
    else if (localName.compare("colorData") == 0)
      stops.append(qMakePair(stopPosition, stopColor));

    return true;
  }
};

TransferFunctionParser::TransferFunctionParser(QString path)
{
  std::cout << "Testing path: " << path.toStdString() << std::endl;
  xmlReader = new QXmlSimpleReader();
  source = new QXmlInputSource(new QFile(path));
  handler = new QGradientContentHandler();
  xmlReader->setContentHandler(handler);
  xmlReader->setErrorHandler(handler);
}

QGradient *TransferFunctionParser::parse()
{
  bool ok = xmlReader->parse(source);

  if (!ok)
    std::cout << "Parsing failed." << std::endl;

  return handler->getQGradientInstance();
}
