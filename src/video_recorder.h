#ifndef SRC_VIDEO_RECORDER_H_

#define SRC_VIDEO_RECORDER_H_

#include <QObject>
#include <QTimer>
#include "./utils/memory.h"
#include "./VolyVideoModule/ffmpegrecorder.h"

class FFMPEGRecorder;
namespace Graphics
{
class Gl;
}

/**
 * \brief
 *
 *
 */
class VideoRecorder : public QObject
{
  Q_OBJECT
 public:
  VideoRecorder(double fps = 24);
  virtual ~VideoRecorder();

  void initialize(Graphics::Gl *gl);
  void resize(int width, int height);

  void createNewVideo(const char *filename);
  void startRecording();
  void stopRecording();
  void updateVideoTimer();
  void captureVideoFrame();

 private:
  double fps;
  std::unique_ptr<FFMPEGRecorder> videoRecorder;
  std::vector<unsigned char> pixelBuffer;
  bool isRecording = false;
  std::unique_ptr<QTimer> videoTimer;
  int videoWidth;
  int videoHeight;
  Graphics::Gl *gl;
};

#endif  // SRC_VIDEO_RECORDER_H_
