#ifndef SRC_VIDEO_RECORDER_H_

#define SRC_VIDEO_RECORDER_H_

#include <QObject>
#include <QTimer>

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
  VideoRecorder() = default;
  virtual ~VideoRecorder();

  void initialize(Graphics::Gl *gl);

  void createVideoRecorder(int xs, int ys, const char *filename,
                           const double fps);
  void startRecording();
  void stopRecording();
  void updateVideoTimer();
  void captureVideoFrame();

 private:
  FFMPEGRecorder *videoRecorder;
  unsigned char *pixelBuffer;
  bool isRecording = false;
  QTimer *videoTimer;
  int videoWidth;
  int videoHeight;
  Graphics::Gl *gl;
};

#endif  // SRC_VIDEO_RECORDER_H_
