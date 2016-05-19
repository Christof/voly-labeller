#ifndef FFMPEG_ENCODER_HPP
#define FFMPEG_ENCODER_HPP

#include "volyvideomoduleAPI.h"
#include <QString>

struct AVCodec;
struct AVCodecContext;
struct AVFrame;
struct SwsContext;

#ifdef __cplusplus
extern "C" {
#endif

#include <libavcodec/avcodec.h>

#ifdef __cplusplus
}
#endif



class VOLYVIDEOMODULE_API ffmpeg_encoder
{
public:
  ffmpeg_encoder(int width, int height, const QString &filename, const double fps);
  virtual ~ffmpeg_encoder();

  void addFrame(unsigned char* buffer);

  static void staticInit();
private:
  int m_width;
  int m_height;
  QString m_filename;
  int m_framecount;
  AVCodec * m_codec;
  AVCodecContext * m_context;
  int m_out_size;
  FILE *m_file;
  AVFrame *m_picture;
  AVPacket m_packet;

  unsigned char *m_rgb_src[3];
  int m_rgb_stride[3];

  SwsContext *m_sws;

  static bool isStaticInit;
};

#endif // FFMPEG_ENCODER_HPP
