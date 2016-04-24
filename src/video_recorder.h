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
 * \brief Provides methods to create a video recording, start and stop it
 *
 * Internaliy it uses the VolyVideoModule, which uses ffmpeg to save videos.
 *
 * The #initialize and #resize methods must called once before using it.
 * #captureVideoFrame must be called once per frame when the completely rendered
 * image is in the back buffer.
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

  void updateVideoTimer();
};

#endif  // SRC_VIDEO_RECORDER_H_
