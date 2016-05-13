#include "ffmpeg_encoder.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#include <libavutil/pixdesc.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>

#ifdef __cplusplus
}
#endif

#include <QDebug>


bool ffmpeg_encoder::isStaticInit = false;

ffmpeg_encoder::ffmpeg_encoder(int width, int height, bool splithorizontal, const QString &filename, const double fps) :
    m_width(width),
    m_height(height),
    m_splithorizontal(splithorizontal),
    m_filename(filename),
    m_framecount(0),
    m_codec(NULL),
    m_context(NULL),
    m_out_size(0),
    m_outbuf_size(1000000),
    m_file(NULL),
    m_picture(NULL),
    m_outbuf(NULL),
    m_picture_buf(NULL)
{
  staticInit();


  /* find the mpeg1 video encoder */
  m_codec = avcodec_find_encoder(CODEC_ID_MPEG2VIDEO);

  if (!m_codec) {
    fprintf(stderr, "codec not found\n");
    ::exit(1);
  }

  m_context = avcodec_alloc_context3(NULL);
#if WIN32
  m_picture = av_frame_alloc();
#else
  m_picture = avcodec_alloc_frame();
#endif


  /* put sample parameters */
  m_context->bit_rate = 100000000;

  /* resolution must be a multiple of two */
  m_context->width = m_width;
  m_context->height = m_height;
  /* frames per second */
  AVRational av;
  av.den = 25;
  av.num = 1;
  m_context->time_base= av;//(AVRational){1,25};
  m_context->gop_size = 10; /* emit one intra frame every ten frames */
  m_context->max_b_frames=0;
  //m_context->qmax = 51;
  //m_context->qmin = 1;

  //the default for mpeg1/2
  m_context->pix_fmt =  PIX_FMT_YUV420P;
  //m_context->pix_fmt =  PIX_FMT_YUV422P;

  char cbuf[12];

  //newer versions of libavcodec have a different API (avcodex_pix_fmt_string is deprecated)

#if (LIBAVCODEC_VERSION_MAJOR <= 52)
  avcodec_pix_fmt_string(cbuf,12, m_context->pix_fmt);
#else
  av_get_pix_fmt_string(cbuf,12, m_context->pix_fmt);
#endif

  printf("AVCODEC STRING %s\n", cbuf);
  if (m_context->pix_fmt < 0)
  {
    fprintf(stderr,"invalid pix format %d",m_context->pix_fmt);
  }

  /* open codec */
  if (avcodec_open2(m_context, m_codec,NULL) < 0) {
    fprintf(stderr, "could not open codec\n");
    ::exit(1);
  }

  /*open file*/
   m_file = fopen(m_filename.toLocal8Bit(), "wb");
  if (!m_file) {
    qDebug() << "could not open " << filename;
    ::exit(1);
  }

  /* alloc image and output buffer */
  m_outbuf = new uint8_t[m_outbuf_size];
  int size = m_context->width * m_context->height;
  m_picture_buf = new uint8_t[(size * 3) / 2]; /* size for YUV 420 */
  //m_picture_buf = new uint8_t[size * 2]; /* size for YUV 422 */

  m_picture->data[0] = m_picture_buf;
  m_picture->data[1] = m_picture->data[0] + size;
  m_picture->data[2] = m_picture->data[1] + size / 4; /* size for YUV 420 */
  //m_picture->data[2] = m_picture->data[1] + size / 2; /* size for YUV 422 */
  m_picture->linesize[0] = m_context->width;
  m_picture->linesize[1] = m_context->width / 2;
  m_picture->linesize[2] = m_context->width / 2;

  m_rgb_src[0] = m_rgb_src[1] = m_rgb_src[2] = NULL;
  m_rgb_stride[0] = -m_context->width*3;
  m_rgb_stride[1] = m_rgb_stride[2] = 0;

  m_sws = sws_getContext(m_context->width, m_context->height, PIX_FMT_RGB24,
                         m_context->width, m_context->height, m_context->pix_fmt,
                         SWS_POINT,
                         NULL, NULL, NULL);
  if (m_sws == NULL)
  {
    fprintf(stderr, "Failed to get %d ---> %d\n",
            PIX_FMT_RGB24,
            m_context->pix_fmt);

    ::exit(1);
  }
}

ffmpeg_encoder::~ffmpeg_encoder()
{

  /* get the delayed frames */
  for(; m_out_size; m_framecount++) {
    fflush(stdout);
    //AVPacket pkt;
    //av_init_packet(&pkt);
    //pkt.data = m_outbuf;
    //pkt.size = m_outbuf_size;
    //m_out_size = avcodec_encode_video2(m_context, m_outbuf);
    m_out_size = avcodec_encode_video(m_context, m_outbuf, m_outbuf_size, NULL);
    printf("write frame %3d (size=%5d)\n", m_framecount, m_out_size);
    fwrite(m_outbuf, 1, m_out_size, m_file);
  }

 /* add sequence end code to have a real mpeg file */
  m_outbuf[0] = 0x00;
  m_outbuf[1] = 0x00;
  m_outbuf[2] = 0x01;
  m_outbuf[3] = 0xb7;
  fwrite(m_outbuf, 1, 4, m_file);
  fclose(m_file);
  delete [] m_picture_buf;
  delete [] m_outbuf;

  avcodec_close(m_context);
  av_free(m_context);
  av_free(m_picture);
  sws_freeContext(m_sws);
}

void ffmpeg_encoder::addFrame(unsigned char* buffer)
{
  // set source pointer
  m_rgb_src[0] = buffer + m_context->width*m_context->height*3 - m_context->width*3;;

  // convert to target pixel format
  sws_scale(m_sws,
            m_rgb_src, m_rgb_stride, 0, m_context->height, // src stuff
            m_picture->data, m_picture->linesize);         // dst stuff

  //printf("%d \n", m_context->height);

  /* encode the image */
  m_out_size = avcodec_encode_video(m_context, m_outbuf, m_outbuf_size, m_picture);

  //printf("encoding frame %3d (size=%5d)\n", m_framecount, m_out_size);

  fwrite(m_outbuf, 1, m_out_size, m_file);

  m_framecount++;
}

void ffmpeg_encoder::staticInit()
{
  if (!isStaticInit)
  {
    /*must be called before using avcodec lib */
    // deprecated avcodec_init();
	
    /* register all the codecs (you can also register only the codec you wish to have smaller code */
    avcodec_register_all();

    isStaticInit = true;
  }
}

