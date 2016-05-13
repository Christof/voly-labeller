#ifndef FFMPEGRECORDER_H
#define FFMPEGRECORDER_H

#include <string>
#include <vector>

#include <QMutex>
#include <QThread>

#include "volyvideomoduleAPI.h"
#include "ffmpegrecorderworker.h"

class FramePool;
class ffmpeg_encoder;

class VOLYVIDEOMODULE_API FFMPEGRecorder
{
public:
  // enumerations
  enum DvrStatus { DVR_READY, DVR_RECORDING, DVR_RECORD_DONE, DVR_STOP_ASKED };

  FFMPEGRecorder(int width, int height, int channels, const std::string &fn, const double fps = 25.0);
  ~FFMPEGRecorder();

  bool isGood() const { return good; }
  std::string errorMessage() const {   return error_message; }
  bool startRecording();
  bool stopRecording();
  void queueFrame(unsigned char* buffer);

  DvrStatus getStatus() const { return status; }

  int nbFramesElapsed() const { return nb_frames_elapsed; }
  int nbFramesStored() const { return nb_frames_stored; }
  int nbFramesLostInCapture() const { return nb_frames_lost_in_capture; }
  int nbFramesLostInEncoding() const { return nb_frames_lost_in_encoding; }
  int nbFramesInEncQueue();
  inline int channels() const { return m_channels; };


private:

  bool good, initialized;
  std::string error_message;

  std::vector<FramePool*> fp;
  std::vector<ffmpeg_encoder*> mpeg_file;
  // semaphores identifiers
  QMutex xmutex ;   // mutex on FFMPEG_file


  // threads identifiers
  std::vector<FFMPEGRecorderWorker*> m_encoding_thread;

  // monitoring

  int nb_frames_elapsed;
  int nb_frames_stored;
  int nb_frames_lost_in_capture;
  int nb_frames_lost_in_encoding;

  int nb_frames_added;

  DvrStatus status;

  // parameters for recording
  int video_width;
  int video_height;
  bool video_splithorizontal;

  double m_fps;

  std::string video_codec_name;

  std::string m_file_name;
  int m_channels;         // 1 for mono, 2 for stereo

  bool endRecording();
  //static void *video_encoding(void *, int i);
  static void wait_for_all_threads_ready(void *);

  friend class FFMPEGRecorderWorker;
};

#endif // FFMPEGRECORDER_H

