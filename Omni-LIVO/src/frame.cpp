/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include <boost/bind.hpp>
#include "feature.h"
#include "frame.h"
#include "visual_point.h"
#include <stdexcept>
#include <vikit/math_utils.h>
#include <vikit/performance_monitor.h>
#include <vikit/vision.h>

int Frame::frame_counter_ = 0;

Frame::Frame(const std::vector<vk::AbstractCamera *> &cams, std::vector<cv::Mat> &imgs)
        : id_(frame_counter_++), cams_(cams), T_f_w_(cams.size(), SE3())
{
    initFrame(imgs);
}

Frame::~Frame()
{
    std::for_each(fts_.begin(), fts_.end(), [&](Feature *i) { delete i; });
}

void Frame::initFrame( std::vector<cv::Mat> &imgs)
{
    if (imgs.empty() || imgs.size() != cams_.size()) {
        throw std::runtime_error("Frame: number of images must match the number of cameras");
    }

    for (size_t i = 0; i < cams_.size(); ++i) {
        if (imgs[i].empty()) {
            throw std::runtime_error("Frame: one of the provided images is empty");
        }

        if (imgs[i].cols != cams_[i]->width() || imgs[i].rows != cams_[i]->height()) {
            throw std::runtime_error("Frame: provided image does not match the size of camera " + std::to_string(i));
        }

        if (imgs[i].type() != CV_8UC1) {
            throw std::runtime_error("Frame: one of the provided images is not grayscale");
        }
    }
    imgs_ = imgs;
}
/// Utility functions for the Frame class
namespace frame_utils
{

void createImgPyramid(const cv::Mat &img_level_0, int n_levels, ImgPyr &pyr)
{
  pyr.resize(n_levels);
  pyr[0] = img_level_0;
  for (int i = 1; i < n_levels; ++i)
  {
    pyr[i] = cv::Mat(pyr[i - 1].rows / 2, pyr[i - 1].cols / 2, CV_8U);
    vk::halfSample(pyr[i - 1], pyr[i]);
  }
}

} // namespace frame_utils
