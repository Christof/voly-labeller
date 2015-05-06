#ifndef SRC_UTILS_PATH_HELPER_H_

#define SRC_UTILS_PATH_HELPER_H_

#include <QFile>
#include <QUrl>
#include <QDir>
#include <QCoreApplication>
#include <string>

inline QString absolutePathOfRelativeUrl(QUrl url)
{
  return QDir(QCoreApplication::applicationDirPath())
      .absoluteFilePath(url.toLocalFile());
}

inline std::string absolutePathOfRelativePath(std::string relativePath)
{
  return QDir(QCoreApplication::applicationDirPath())
      .absoluteFilePath(QString(relativePath.c_str()))
      .toStdString();
}

#endif  // SRC_UTILS_PATH_HELPER_H_
