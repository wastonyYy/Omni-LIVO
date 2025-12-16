#ifndef LIVO_FRAME_H_
#define LIVO_FRAME_H_

#include <boost/noncopyable.hpp>
#include <vikit/abstract_camera.h>
#include <vector>
#include <list>

class VisualPoint;
struct Feature;

typedef std::list<Feature *> Features;
typedef std::vector<cv::Mat> ImgPyr;

class Frame : boost::noncopyable
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static int frame_counter_;
    int id_;
    std::vector<vk::AbstractCamera *> cams_; // Multiple cameras
    std::vector<SE3> T_f_w_;                 // Transform (f)rame from (w)orld for each camera
    std::vector<SE3> T_f_w_prior_;          //!< Transform (f)rame from (w)orld provided by the IMU prior.
    std::vector<cv::Mat> imgs_;
    Features fts_;

    Frame(const std::vector<vk::AbstractCamera *> &cams, std::vector<cv::Mat> &imgs);
    ~Frame();

    void initFrame( std::vector<cv::Mat> &imgs);

    inline size_t nObs() const { return fts_.size(); }

    // Transforms point coordinates in world-frame (w) to camera pixel coordinates (c).

    inline Vector2d w2c(const Vector3d& xyz_w, int cam_id) const {
        return cams_[cam_id]->world2cam(T_f_w_[cam_id] * xyz_w);
    }
    inline Vector2d w2c_prior(const Vector3d& xyz_w, int cam_id) const {
        return cams_[cam_id]->world2cam(T_f_w_prior_[cam_id] * xyz_w);
    }
    inline Vector3d f2w(const Vector3d& f, int cam_idx) const {
        return T_f_w_[cam_idx].inverse() * f;
    }
    inline Vector3d c2f(const double x, const double y, int cam_idx) const {
        return cams_[cam_idx]->cam2world(x, y);
    }

    inline Vector3d w2f(const Vector3d &xyz_w, int cam_idx) const {
        return T_f_w_[cam_idx] * xyz_w;
    }

    inline Vector3d pos(int cam_idx) const {
        return T_f_w_[cam_idx].inverse().translation();
    }
};

typedef std::unique_ptr<Frame> FramePtr;

namespace frame_utils
{
    void createImgPyramid(const cv::Mat &img_level_0, int n_levels, ImgPyr &pyr);
} // namespace frame_utils

#endif // LIVO_FRAME_H_
