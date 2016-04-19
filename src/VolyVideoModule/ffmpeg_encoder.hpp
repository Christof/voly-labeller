#ifndef FFMPEG_ENCODER_HPP
#define FFMPEG_ENCODER_HPP

#include "volyvideomoduleAPI.h"
#include <QString>

struct AVCodec;
struct AVCodecContext;
struct AVFrame;
struct SwsContext;

class VOLYVIDEOMODULE_API ffmpeg_encoder
{
public:
  ffmpeg_encoder(int width, int height, bool splithorizontal, const QString &filename, const double fps);
  virtual ~ffmpeg_encoder();

  void addFrame(unsigned char* buffer);

  static void staticInit();
private:
  int m_width;
  int m_height;
  bool m_splithorizontal;
  QString m_filename;
  int m_framecount;
  AVCodec * m_codec;
  AVCodecContext * m_context;
  int m_out_size;
  int m_outbuf_size;
  FILE *m_file;
  AVFrame *m_picture;
  unsigned char *m_outbuf;

  unsigned char *m_picture_buf;
  int m_picture_stride[3];

  unsigned char *m_rgb_src[3];
  int m_rgb_stride[3];

  SwsContext *m_sws;

  static bool isStaticInit;
};

#endif // FFMPEG_ENCODER_HPP
