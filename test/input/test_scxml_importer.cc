#include "../test.h"
#include <QUrl>
#include "../../src/input/scxml_importer.h"

TEST(Test_ScxmlImporter, foo)
{
  ScxmlImporter importer(QUrl::fromLocalFile("../config/simple_state.xml"));
}
