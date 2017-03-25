#ifndef SRC_UTILS_TRANSFER_FUNCTION_PARSER_H_

#define SRC_UTILS_TRANSFER_FUNCTION_PARSER_H_

#include <QGradient>
#include <QString>
#include <QXmlDefaultHandler>

class QGradientContentHandler;

/**
 * \brief Parser for XML representation of a QGradient used as transfer function
 *
 */
class TransferFunctionParser
{
 public:
  explicit TransferFunctionParser(QString path);

  QGradient *parse();

 private:
  QXmlSimpleReader *xmlReader;
  QXmlInputSource *source;
  QGradientContentHandler *handler;
};

#endif  // SRC_UTILS_TRANSFER_FUNCTION_PARSER_H_
