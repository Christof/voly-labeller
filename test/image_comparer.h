#ifndef TEST_IMAGE_COMPARER_H_

#define TEST_IMAGE_COMPARER_H_

#include <QImage>
#include <QFile>

void compareImages(const char *expectedPath, const char *actualPath,
                   const QImage *actualImage)
{
  actualImage->save(actualPath);

  QFile expectedFile(expectedPath);
  ASSERT_TRUE(expectedFile.exists())
      << "File '" << expectedPath << "' does not exists. Check '" << actualPath
      << "' and rename it if it is correct.";

  QImage expectedImage(expectedFile.fileName());
  ASSERT_EQ(expectedImage.width(), actualImage->width());
  ASSERT_EQ(expectedImage.height(), actualImage->height());

  for (int y = 0; y < expectedImage.width(); ++y)
    for (int x = 0; x < expectedImage.width(); ++x)
      EXPECT_EQ(expectedImage.pixel(x, y), actualImage->pixel(x, y));
}

void compareImages(const char *expectedPath, const char *actualPath)
{
  QImage actualImage(actualPath);
  compareImages(expectedPath, actualPath, &actualImage);
}

#endif  // TEST_IMAGE_COMPARER_H_
