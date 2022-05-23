#ifndef GREY_CUH
#define GREY_CUH
double grey_filter(unsigned char* host_dst, unsigned char* host_src, int width, int height);
double grey_filter_Optimized(unsigned char* host_dst, unsigned char* host_src, int width, int height);
#endif //GREY_CUH
