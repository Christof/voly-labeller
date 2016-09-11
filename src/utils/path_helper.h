#ifndef SRC_UTILS_PATH_HELPER_H_

#define SRC_UTILS_PATH_HELPER_H_

#include <QFile>
#include <QUrl>
#include <QDir>
#include <QCoreApplication>
#include <string>
#include "./project_root.h"

inline QString relativeApplicationToProjectRootPath()
{
  return QDir(QCoreApplication::applicationDirPath())
      .relativeFilePath(QString(PROJECT_ROOT));
}

inline QString absolutePathToProjectRelativePath(QString filename)
{
  return QDir(QString(PROJECT_ROOT)).relativeFilePath(filename);
}

inline QString absolutePathOfRelativeUrl(QUrl url)
{
  return QDir(QCoreApplication::applicationDirPath())
      .absoluteFilePath(url.toLocalFile());
}

inline QString absolutePathOfProjectRelativePath(QString path)
{
  return QDir(QDir(QCoreApplication::applicationDirPath())
                  .absoluteFilePath(relativeApplicationToProjectRootPath()))
      .absoluteFilePath(path);
}

inline QString absolutePathOfProjectRelativeUrl(QUrl url)
{
  return absolutePathOfProjectRelativePath(url.toLocalFile());
}

inline std::string absolutePathOfRelativePath(std::string relativePath)
{
  return QDir(QCoreApplication::applicationDirPath())
      .absoluteFilePath(QString(relativePath.c_str()))
      .toStdString();
}

inline std::string absolutePathOfProjectRelativePath(std::string relativePath)
{
  return absolutePathOfProjectRelativePath(QString(relativePath.c_str()))
      .toStdString();
}

inline std::string replaceBackslashesWithSlashes(std::string path)
{
  std::replace(path.begin(), path.end(), '\\', '/');
  return path;
}

inline std::string filenameWithoutExtension(std::string path)
{
  return QFileInfo(path.c_str()).baseName().toStdString();
}

#endif  // SRC_UTILS_PATH_HELPER_H_
