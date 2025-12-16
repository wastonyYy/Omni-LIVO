
#include "vio.h"

VIOManager::VIOManager() {
    // downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
}

VIOManager::~VIOManager() {
    delete visual_submap;
    for (auto &pair: warp_map) delete pair.second;
    warp_map.clear();
    for (auto &pair: feat_map) delete pair.second;
    feat_map.clear();
}

void VIOManager::setImuToLidarExtrinsic(const V3D &transl, const M3D &rot) {
    Pli = -rot.transpose() * transl;
    Rli = rot.transpose();
}

void VIOManager::setLidarToCameraExtrinsic(std::vector<std::vector<double>> &R,
                                           std::vector<std::vector<double>> &P) {
    // R.size() == P.size() == cams.size()
    Rcl_vec.resize(cams.size());
    Pcl_vec.resize(cams.size());
    for (size_t i = 0; i < cams.size(); i++) {
        Rcl_vec[i] << R[i][0], R[i][1], R[i][2],
                R[i][3], R[i][4], R[i][5],
                R[i][6], R[i][7], R[i][8];
        Pcl_vec[i] << P[i][0], P[i][1], P[i][2];
    }
}

/**
 * @brief 初始化VIO
 *  - 这里的多相机分辨率都一致，因此统一使用 width/height
 */
void VIOManager::initializeVIO() {
    // 1) 检查至少有一个相机
    if (cams.empty()) {
        std::cerr << "[VIOManager::initializeVIO] Error: no cameras!\n";
        return;
    }
    // 2) 读取第0个相机的 width/height 作为标准
    width = cams[0]->width();
    height = cams[0]->height();

    // 3) 确保所有相机分辨率都与第0个相机一致
    for (size_t i = 1; i < cams.size(); i++) {
        if (cams[i]->width() != width || cams[i]->height() != height) {
            ROS_ERROR("[initializeVIO()] cameras has different resolution");
            return;
        }
    }
    visual_submap = new SubSparseMap;
    image_resize_factor = cams[0]->scale();

    printf("width: %d, height: %d, scale: %f\n", width, height, image_resize_factor);

    Rci_vec.resize(cams.size());
    Rcl_vec.resize(cams.size());
    Rcw_vec.resize(cams.size());
    Pci_vec.resize(cams.size());
    Pcl_vec.resize(cams.size());
    Pcw_vec.resize(cams.size());

    Jdphi_dR_vec.resize(cams.size());
    Jdp_dt_vec.resize(cams.size());
    Jdp_dR_vec.resize(cams.size());

    for(size_t i = 0; i < cams.size(); i++)
    {
        Rci_vec[i] = Rcl_vec[i] * Rli;
        Pci_vec[i] = Rcl_vec[i] * Pli + Pcl_vec[i];
        Jdphi_dR_vec[i] = Rci_vec[i];
        Eigen::Vector3d Pic_i = - Rci_vec[i].transpose() * Pci_vec[i];

        // SKEW_SYM_MATRX(Pic) => 构建反对称矩阵
        Eigen::Matrix3d tmp;
        tmp << 0,        -Pic_i.z(),  Pic_i.y(),
                Pic_i.z(),  0,        -Pic_i.x(),
                -Pic_i.y(),  Pic_i.x(), 0;

        Jdp_dR_vec[i] = - Rci_vec[i] * tmp;

    }
    if (grid_size > 10) {
        grid_n_width = ceil(static_cast<double>(width / grid_size));
        grid_n_height = ceil(static_cast<double>(height / grid_size));
    } else {
        grid_size = static_cast<int>(height / grid_n_height);
        grid_n_height = ceil(static_cast<double>(height / grid_size));
        grid_n_width = ceil(static_cast<double>(width / grid_size));
    }
    length = grid_n_width * grid_n_height;

    if (colmap_output_en) {
        pinhole_cam = dynamic_cast<vk::PinholeCamera *>(cams[0]);
        fout_colmap.open(DEBUG_FILE_DIR("Colmap/sparse/0/images.txt"), ios::out);
        fout_colmap << "# Image list with two lines of data per image:\n";
        fout_colmap << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
        fout_colmap << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
        fout_camera.open(DEBUG_FILE_DIR("Colmap/sparse/0/cameras.txt"), ios::out);
        fout_camera << "# Camera list with one line of data per camera:\n";
        fout_camera << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
        fout_camera << "1 PINHOLE " << width << " " << height << " "
                    << std::fixed << std::setprecision(6)  // 控制浮点数精度为10位
                    << cams[0]->fx() << " " << cams[0]->fy() << " "
                    << cams[0]->cx() << " " << cams[0]->cy() << std::endl;
        fout_camera.close();
    }
    grid_num.resize(length);
    map_index.resize(length);
    map_dist.resize(length);
    update_flag.resize(length);
    scan_value.resize(length);

    patch_size_total = patch_size * patch_size;
    patch_size_half = static_cast<int>(patch_size / 2);
    patch_buffer.resize(patch_size_total);
    warp_len = patch_size_total * patch_pyrimid_level;
    border = (patch_size_half + 2) * 8;

    retrieve_voxel_points.reserve(length);
    append_voxel_points.reserve(length);

    sub_feat_map.clear();
}

void VIOManager::resetGrid() {
    fill(grid_num.begin(), grid_num.end(), TYPE_UNKNOWN);
    fill(map_index.begin(), map_index.end(), 0);
    fill(map_dist.begin(), map_dist.end(), 10000.0f);
    fill(update_flag.begin(), update_flag.end(), 0);
    fill(scan_value.begin(), scan_value.end(), 0.0f);

    retrieve_voxel_points.clear();
    retrieve_voxel_points.resize(length);

    append_voxel_points.clear();
    append_voxel_points.resize(length);

    total_points = 0;
}

// void VIOManager::resetRvizDisplay()
// {
// sub_map_ray.clear();
// sub_map_ray_fov.clear();
// visual_sub_map_cur.clear();
// visual_converged_point.clear();
// map_cur_frame.clear();
// sample_points.clear();
// }



void VIOManager::computeProjectionJacobian(int cam_idx, V3D p, MD(2, 3) &J)
{
    double fx_i = cams[cam_idx]->fx();
    double fy_i = cams[cam_idx]->fy();

    // 取出三维点 p = (x, y, z) (相机坐标系下)
    const double x = p[0];
    const double y = p[1];
    const double z_inv = 1.0 / p[2];
    const double z_inv_2 = z_inv * z_inv;

    J(0, 0) = fx_i * z_inv;   // d(px)/d(x)
    J(0, 1) = 0.0;            // d(px)/d(y)
    J(0, 2) = -fx_i * x * z_inv_2;  // d(px)/d(z)

    J(1, 0) = 0.0;
    J(1, 1) = fy_i * z_inv;
    J(1, 2) = -fy_i * y * z_inv_2;
}


void VIOManager::getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level) {
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int scale = (1 << level);
    const int u_ref_i = floorf(pc[0] / scale) * scale;
    const int v_ref_i = floorf(pc[1] / scale) * scale;
    const float subpix_u_ref = (u_ref - u_ref_i) / scale;
    const float subpix_v_ref = (v_ref - v_ref_i) / scale;
    const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    for (int x = 0; x < patch_size; x++) {
        uint8_t *img_ptr = (uint8_t *) img.data + (v_ref_i - patch_size_half * scale + x * scale) * width +
                           (u_ref_i - patch_size_half * scale);
        for (int y = 0; y < patch_size; y++, img_ptr += scale) {
            patch_tmp[patch_size_total * level + x * patch_size + y] =
                    w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] +
                    w_ref_br * img_ptr[scale * width + scale];
        }
    }
}

void VIOManager::insertPointIntoVoxelMap(VisualPoint *pt_new) {
    V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
    double voxel_size = 0.5;
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
        loc_xyz[j] = pt_w[j] / voxel_size;
        if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t) loc_xyz[0], (int64_t) loc_xyz[1], (int64_t) loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
        iter->second->voxel_points.push_back(pt_new);
        iter->second->count++;
    } else {
        VOXEL_POINTS *ot = new VOXEL_POINTS(0);
        ot->voxel_points.push_back(pt_new);
        feat_map[position] = ot;
    }
}

void VIOManager::getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref, const V3D &xyz_ref,
                                               const V3D &normal_ref,
                                               const SE3 &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref) {
    // create homography matrix
    const V3D t = T_cur_ref.inverse().translation();
    const Eigen::Matrix3d H_cur_ref =
            T_cur_ref.rotation_matrix() *
            (normal_ref.dot(xyz_ref) * Eigen::Matrix3d::Identity() - t * normal_ref.transpose());
    // Compute affine warp matrix A_ref_cur using homography projection
    const int kHalfPatchSize = 4;
    V3D f_du_ref(cam.cam2world(px_ref + Eigen::Vector2d(kHalfPatchSize, 0) * (1 << level_ref)));
    V3D f_dv_ref(cam.cam2world(px_ref + Eigen::Vector2d(0, kHalfPatchSize) * (1 << level_ref)));
    //   f_du_ref = f_du_ref/f_du_ref[2];
    //   f_dv_ref = f_dv_ref/f_dv_ref[2];
    const V3D f_cur(H_cur_ref * xyz_ref);
    const V3D f_du_cur = H_cur_ref * f_du_ref;
    const V3D f_dv_cur = H_cur_ref * f_dv_ref;
    V2D px_cur(cam.world2cam(f_cur));
    V2D px_du_cur(cam.world2cam(f_du_cur));
    V2D px_dv_cur(cam.world2cam(f_dv_cur));
    A_cur_ref.col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
    A_cur_ref.col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

void VIOManager::getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref,
                                     const double depth_ref,
                                     const SE3 &T_cur_ref, const int level_ref, const int pyramid_level,
                                     const int halfpatch_size,
                                     Matrix2d &A_cur_ref) {
    // Compute affine warp matrix A_ref_cur
    const Vector3d xyz_ref(f_ref * depth_ref);
    Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size, 0) * (1 << level_ref) * (1 << pyramid_level)));
    Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0, halfpatch_size) * (1 << level_ref) * (1 << pyramid_level)));
    xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
    xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];
    const Vector2d px_cur(cam.world2cam(T_cur_ref * (xyz_ref)));
    const Vector2d px_du(cam.world2cam(T_cur_ref * (xyz_du_ref)));
    const Vector2d px_dv(cam.world2cam(T_cur_ref * (xyz_dv_ref)));
    A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
    A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
}

void
VIOManager::warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref,
                       const int search_level,
                       const int pyramid_level, const int halfpatch_size, float *patch) {
    const int patch_size = halfpatch_size * 2;
    const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
    if (isnan(A_ref_cur(0, 0))) {
        printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
        return;
    }

    float *patch_ptr = patch;
    for (int y = 0; y < patch_size; ++y) {
        for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
        {
            Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);
            px_patch *= (1 << search_level);
            px_patch *= (1 << pyramid_level);
            const Vector2f px(A_ref_cur * px_patch + px_ref.cast<float>());
            if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1)
                patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = 0;
            else
                patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = (float) vk::interpolateMat_8u(
                        img_ref, px[0], px[1]);
        }
    }
}

int VIOManager::getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level) {
    // Compute patch level in other image
    int search_level = 0;
    double D = A_cur_ref.determinant();
    while (D > 3.0 && search_level < max_level) {
        search_level += 1;
        D *= 0.25;
    }
    return search_level;
}

double VIOManager::calculateNCC(float *ref_patch, float *cur_patch, int patch_size) {
    double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
    double mean_ref = sum_ref / patch_size;

    double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
    double mean_curr = sum_cur / patch_size;

    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < patch_size; i++) {
        double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
        numerator += n;
        demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
        demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

void VIOManager::retrieveFromVisualSparseMap(const std::vector<cv::Mat> imgs,
                                             vector<pointWithVar> &pg,
                                             const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
    // 如果当前没有任何 feature map，则直接返回
    if (feat_map.empty())
        return;

    double ts0 = omp_get_wtime();

    // 重置可视化子地图信息
    visual_submap->reset();
    sub_feat_map.clear();

    float voxel_size = 0.5f;

    // 如果 normal_en 为 false，则清空 warp_map
    if (!normal_en)
        warp_map.clear();

    // 为每台相机分配一张深度图（假设各相机分辨率相同）
    std::vector<cv::Mat> depth_imgs(cams.size());
    for (size_t i = 0; i < cams.size(); i++)
    {
        depth_imgs[i] = cv::Mat::zeros(height, width, CV_32FC1);
    }

    grid_n_width = width / grid_size;
    grid_n_height = height / grid_size;
    int total_cells = grid_n_width * grid_n_height;
    grid_num.assign(total_cells, 0);
    map_dist.assign(total_cells, std::numeric_limits<float>::max());
    retrieve_voxel_points.assign(total_cells, nullptr);
    //-------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // 1) 将 pg 中的 3D 点投影到每个相机，并写到对应的 depth_imgs 中
    //--------------------------------------------------------------------------
#ifdef DEBUG_TIME
    double t_insert = 0.0, t_depth = 0.0, t_position = 0.0;
#endif

    for (int i_point = 0; i_point < (int)pg.size(); i_point++)
    {
#ifdef DEBUG_TIME
        double t0 = omp_get_wtime();
#endif
        V3D pt_w = pg[i_point].point_w;
        int loc_xyz[3];

        // 计算离散体素位置
        for (int j = 0; j < 3; j++)
        {
            loc_xyz[j] = (int)std::floor(pt_w[j] / voxel_size);
            if (loc_xyz[j] < 0)
                loc_xyz[j] -= 1;
        }
        VOXEL_LOCATION position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

#ifdef DEBUG_TIME
        double t1 = omp_get_wtime();
        t_position += (t1 - t0);
#endif
        // 将该 voxel 注册到 sub_feat_map（重复注册时直接重置为 0）
        sub_feat_map[position] = 0;

#ifdef DEBUG_TIME
        double t2 = omp_get_wtime();
        t_insert += (t2 - t1);
#endif

        // 针对每台相机进行投影
        for (int cam_idx = 0; cam_idx < (int)cams.size(); cam_idx++)
        {
            V3D pt_c = new_frame_->w2f(pt_w, cam_idx);
            if (pt_c[2] > 0)
            {
                V2D px = new_frame_->w2c(pt_w, cam_idx);
                if (cams[cam_idx]->isInFrame(px.cast<int>(), border))
                {
                    float depth = (float)pt_c[2];
                    int col = (int)px[0];
                    int row = (int)px[1];
                    float *it = (float*)depth_imgs[cam_idx].data;
                    it[row * width + col] = depth;
                }
            }
        }
#ifdef DEBUG_TIME
        double t3 = omp_get_wtime();
        t_depth += (t3 - t2);
#endif
    } // end for pg.size()

#ifdef DEBUG_TIME
    std::cout << "[ Debug time ] calculate pt position: " << t_position << " s" << std::endl;
    std::cout << "[ Debug time ] sub_postion.insert(position): " << t_insert << " s" << std::endl;
    std::cout << "[ Debug time ] generate depth map: " << t_depth << " s" << std::endl;
#endif

    //--------------------------------------------------------------------------
    // 2) 遍历 sub_feat_map ，在 feat_map 中查找对应的 voxel，判断是否在任一相机视野中
    //--------------------------------------------------------------------------
    std::vector<VOXEL_LOCATION> DeleteKeyList;
    for (auto &iter_sf : sub_feat_map)
    {
        VOXEL_LOCATION position = iter_sf.first;
        auto corre_voxel = feat_map.find(position);
        if (corre_voxel != feat_map.end())
        {
            bool voxel_in_fov = false; // 是否被任一相机看到
            std::vector<VisualPoint*> &voxel_points = corre_voxel->second->voxel_points;

            for (VisualPoint* pt : voxel_points)
            {
                if (!pt || pt->obs_.empty())
                    continue;
                bool this_pt_in_fov = false;

                for (int cam_idx = 0; cam_idx < (int)cams.size(); cam_idx++)
                {
                    V3D dir_c = new_frame_->w2f(pt->pos_, cam_idx);
                    if (dir_c[2] < 0)
                        continue;

                    V2D pc = new_frame_->w2c(pt->pos_, cam_idx);
                    if (!cams[cam_idx]->isInFrame(pc.cast<int>(), border))
                        continue;

                    this_pt_in_fov = true;

                    // 计算所在网格索引（这里按 cam0 尺寸计算）
                    int grid_col = (int)(pc[0] / grid_size);
                    int grid_row = (int)(pc[1] / grid_size);
                    int index = grid_row * grid_n_width + grid_col;
                    if (index < 0 || index >= total_cells)
                        continue;

                    grid_num[index] = TYPE_MAP;

                    // 计算观测距离，记录最小距离
                    V3D obs_vec = new_frame_->pos(cam_idx) - pt->pos_;
                    float cur_dist = (float)obs_vec.norm();
                    if (cur_dist <= map_dist[index])
                    {
                        map_dist[index] = cur_dist;
                        retrieve_voxel_points[index] = pt;
                    }
                }

                if (this_pt_in_fov)
                    voxel_in_fov = true;
            }

            if (!voxel_in_fov)
                DeleteKeyList.push_back(position);
        }
    }
    // 删除不在视野中的 voxel
    for (auto &key : DeleteKeyList)
    {
        sub_feat_map.erase(key);
    }

    // 修改后的多相机 patch 提取部分（原来只用 cam0，现在遍历所有相机）
    for (int i = 0; i < total_cells; i++)
    {
        if (grid_num[i] != TYPE_MAP)
            continue;

        VisualPoint* pt = retrieve_voxel_points[i];
        if (!pt)
            continue;

        // 记录多相机中最优观测的相关信息
        float best_error = std::numeric_limits<float>::max();
        int best_cam = -1;
        int best_search_level = 0;
        Matrix2d best_A_cur_ref_zero;
        std::vector<float> best_patch_wrap;
        best_patch_wrap.resize(warp_len);
        Feature* best_ref_ftr = nullptr;
        V2D best_pc;  // 最优相机下的像素坐标

        // 遍历所有相机
        for (int cam_idx = 0; cam_idx < (int)cams.size(); cam_idx++)
        {
            // 投影当前点到当前相机
            V2D pc = new_frame_->w2c(pt->pos_, cam_idx);

            // 检查是否足够远离图像边界，确保 patch 提取区域有效
            if (pc[0] < patch_size_half || pc[0] >= width - patch_size_half ||
                pc[1] < patch_size_half || pc[1] >= height - patch_size_half)
                continue;
            if (!cams[cam_idx]->isInFrame(pc.cast<int>(), border))
                continue;

            // 深度连续性检查（使用当前相机对应的 depth_imgs[cam_idx]）
            V3D pt_cam = new_frame_->w2f(pt->pos_, cam_idx);
            bool depth_continous = false;
            for (int u = -patch_size_half; u <= patch_size_half && !depth_continous; u++) {
                for (int v = -patch_size_half; v <= patch_size_half; v++) {
                    if (u == 0 && v == 0)
                        continue;
                    int col = (int)pc[0] + u;
                    int row = (int)pc[1] + v;
                    if (col < 0 || col >= width || row < 0 || row >= height)
                        continue;
                    float *depth_ptr = reinterpret_cast<float*>(depth_imgs[cam_idx].data);
                    float depth = depth_ptr[row * width + col];
                    if (depth == 0.f)
                        continue;
                    double delta_dist = std::fabs(pt_cam[2] - depth);
                    if (delta_dist > 0.5) {
                        depth_continous = true;
                        break;
                    }
                }
            }
            if (depth_continous)
                continue;

            // 针对当前相机，选取或确定参考特征
            Feature *ref_ftr = nullptr;
            if (normal_en)
            {
                float photometric_errors_min = std::numeric_limits<float>::max();
                if (pt->obs_.size() == 1)
                {
                    ref_ftr = *pt->obs_.begin();
                    pt->ref_patch = ref_ftr;
                    pt->has_ref_patch_ = true;
                }
                else if (!pt->has_ref_patch_)
                {
                    for (auto it = pt->obs_.begin(); it != pt->obs_.end(); ++it)
                    {
                        Feature* ref_patch_temp = *it;
                        float *patch_temp = ref_patch_temp->patch_;
                        float photometric_errors = 0.0f;
                        int count = 0;
                        for (auto itm = pt->obs_.begin(); itm != pt->obs_.end(); ++itm)
                        {
                            if ((*itm)->id_ == ref_patch_temp->id_)
                                continue;
                            float *patch_cache = (*itm)->patch_;
                            for (int ind = 0; ind < patch_size_total; ind++)
                            {
                                float diff = (patch_temp[ind] - patch_cache[ind]);
                                photometric_errors += diff * diff;
                            }
                            count++;
                        }
                        if (count > 0)
                            photometric_errors /= (float)count;
                        if (photometric_errors < photometric_errors_min)
                        {
                            photometric_errors_min = photometric_errors;
                            ref_ftr = ref_patch_temp;
                        }
                    }
                    pt->ref_patch = ref_ftr;
                    pt->has_ref_patch_ = true;
                }
                else
                {
                    ref_ftr = pt->ref_patch;
                }
            }
            else
            {
                // 当 normal_en 为 false 时，选择距离当前相机最近的观测
                if (!pt->getCloseViewObs(new_frame_->pos(cam_idx), ref_ftr, pc))
                    continue;
            }
            if (!ref_ftr)
                continue;

            // 构造 warp 参数（使用当前相机的参数）
            int search_level;
            Matrix2d A_cur_ref_zero;
            if (normal_en)
            {
                V3D norm_vec = (ref_ftr->T_f_w_.rotation_matrix() * pt->normal_).normalized();
                V3D pf = ref_ftr->T_f_w_ * pt->pos_;
                SE3 T_cur_ref = new_frame_->T_f_w_[cam_idx] * ref_ftr->T_f_w_.inverse();
                getWarpMatrixAffineHomography(*cams[cam_idx], ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref_zero);
                search_level = getBestSearchLevel(A_cur_ref_zero, 2);
            }
            else
            {
                auto iter_warp = warp_map.find(ref_ftr->id_);
                if (iter_warp != warp_map.end())
                {
                    search_level = iter_warp->second->search_level;
                    A_cur_ref_zero = iter_warp->second->A_cur_ref;
                }
                else
                {
                    getWarpMatrixAffine(*cams[cam_idx], ref_ftr->px_, ref_ftr->f_,
                                        (ref_ftr->pos() - pt->pos_).norm(),
                                        new_frame_->T_f_w_[cam_idx] * ref_ftr->T_f_w_.inverse(),
                                        ref_ftr->level_, 0, patch_size_half, A_cur_ref_zero);
                    search_level = getBestSearchLevel(A_cur_ref_zero, 2);
                    Warp *ot = new Warp(search_level, A_cur_ref_zero);
                    warp_map[ref_ftr->id_] = ot;
                }
            }

            // 执行 warp 操作，提取 patch（使用当前相机的参考图像）
            std::vector<float> patch_wrap(warp_len);
            for (int pyramid_level = 0; pyramid_level <= patch_pyrimid_level - 1; pyramid_level++)
            {
                warpAffine(A_cur_ref_zero, ref_ftr->img_, ref_ftr->px_, ref_ftr->level_,
                           search_level, pyramid_level, patch_size_half, patch_wrap.data());
            }

            // 从当前相机图像提取 patch
            std::vector<float> patch_buffer_local(patch_size_total);
            getImagePatch(imgs[cam_idx], pc, patch_buffer_local.data(), 0);

            // 计算光度误差
            float error = 0.0f;
            for (int ind = 0; ind < patch_size_total; ind++)
            {
                float diff = ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer_local[ind];
                error += diff * diff;
            }

            if (ncc_en)
            {
                double ncc = calculateNCC(patch_wrap.data(), patch_buffer_local.data(), patch_size_total);
                if (ncc < ncc_thre)
                    continue;
            }

            if (error > outlier_threshold * patch_size_total)
                continue;

            // 若当前相机观测误差更好，则记录下来
            if (error < best_error)
            {
                best_error = error;
                best_cam = cam_idx;
                best_search_level = search_level;
                best_A_cur_ref_zero = A_cur_ref_zero;
                best_patch_wrap = patch_wrap;
                best_ref_ftr = ref_ftr;
                best_pc = pc;
            }
        } // end for each cam_idx

        // 如果至少有一个相机提供了有效观测，则使用最佳观测更新 visual_submap
        if (best_cam != -1)
        {
            visual_submap->voxel_points.push_back(pt);
            visual_submap->propa_errors.push_back(best_error);
            visual_submap->search_levels.push_back(best_search_level);
            visual_submap->errors.push_back(best_error);
            visual_submap->warp_patch.push_back(best_patch_wrap);
            visual_submap->inv_expo_list.push_back(best_ref_ftr->inv_expo_time_);
        }
    }
    total_points = (int)visual_submap->voxel_points.size();
    double ts1 = omp_get_wtime();
    printf("[ VIO ] Retrieve %d points from visual sparse map (multi-cam). cost=%.6lf s\n",
           total_points, ts1 - ts0);

}

void VIOManager::computeJacobianAndUpdateEKF(const std::vector<cv::Mat> imgs)
{
    if (total_points == 0) {
        ROS_WARN("[computeJacobianAndUpdateEKF] total_points == 0");
    };


    compute_jacobian_time = update_ekf_time = 0.0;

    for (int level = patch_pyrimid_level - 1; level >= 0; level--)
    {
        if (inverse_composition_en)
        {
            ROS_WARN("[updateStateInverse]  inverse_composition_en");
            has_ref_patch_cache = false;
            updateStateInverse(imgs, level);
        }
        else{
            updateState(imgs, level);
        }
    }
    state->cov -= G * state->cov;
    updateFrameState(*state);
}

void VIOManager::generateVisualMapPoints(const std::vector<cv::Mat>& imgs, std::vector<pointWithVar> &pg)
{
    // 如果外部点太少，则直接返回
    if (pg.size() <= 10)
        return;
    std::vector<int> best_cam_idx(length, -1);//重合点取最好的相机


    // 对每个点遍历所有相机，选取评分最高的投影
    for (size_t i = 0; i < pg.size(); i++) {
        // 如果该点没有有效法向量，则跳过
        if (pg[i].normal == V3D(0, 0, 0))
            continue;

        V3D pt = pg[i].point_w;
        // 对于每个相机
        for (size_t cam_idx = 0; cam_idx < imgs.size(); cam_idx++) {
            // 投影：调用新帧的多相机接口
            V2D pc = new_frame_->w2c(pt, cam_idx);
            // 检查该投影是否在当前相机图像内（考虑 border 边界）
            if (!cams[cam_idx]->isInFrame(pc.cast<int>(), border))
                continue;

            // 计算该投影所属的网格索引（这里假设网格参数对所有相机相同）
            int grid_col = static_cast<int>(pc[0] / grid_size);
            int grid_row = static_cast<int>(pc[1] / grid_size);
            int index = grid_row * grid_n_width + grid_col;

            // 如果该网格已经属于地图点（TYPE_MAP），则不再更新
            if (grid_num[index] == TYPE_MAP)
                continue;

            // 计算当前图像中该像素的质量得分（例如 Shi-Tomasi 分数）
            float cur_value = vk::shiTomasiScore(imgs[cam_idx], pc[0], pc[1]);

            // 如果得分高于当前该网格记录的分数，则更新
            if (cur_value > scan_value[index]) {
                scan_value[index] = cur_value;
                append_voxel_points[index] = pg[i]; // 存储该 3D 点的信息
                best_cam_idx[index] = static_cast<int>(cam_idx);
                grid_num[index] = TYPE_POINTCLOUD;
            }
        } // end for each camera
    } // end for each pg point

    // 接下来遍历整个网格，将标记为 TYPE_POINTCLOUD 的生成新的地图点
    int add_count = 0;
    for (int i = 0; i < length; i++) {
        if (grid_num[i] != TYPE_POINTCLOUD)
            continue;

        // 从对应网格中取出 3D 点信息
        pointWithVar pt_var = append_voxel_points[i];
        V3D pt = pt_var.point_w;
        // 使用对应的最佳相机索引
        int cam_idx = best_cam_idx[i];
        // 投影到图像平面
        V2D pc = new_frame_->w2c(pt, cam_idx);

        // 这里可以根据需要检查视角、夹角等（例如判断与法线的夹角）
        // 例如：计算点在相机坐标下的深度
        V3D pt_cam = new_frame_->w2f(pt, cam_idx);
        if (pt_cam[2] <= 0)
            continue;

        // 从对应相机图像中提取 patch（这里 level 取 0，即原图）
        float *patch = new float[patch_size_total];
        getImagePatch(imgs[cam_idx], pc, patch, 0);

        // 创建新的 VisualPoint 对象
        VisualPoint *pt_new = new VisualPoint(pt);
        // 计算该点对应的方向向量 f（例如通过 cam2world）
        Vector3d f = cams[cam_idx]->cam2world(pc);
        // 创建新的 Feature 对象，传入当前相机位姿 new_frame_->T_f_w_[cam_idx]
        Feature *ftr_new = new Feature(pt_new, patch, pc, f, new_frame_->T_f_w_[cam_idx], 0, cam_idx);
        ftr_new->cam_id_ = cam_idx;
        ftr_new->img_ = imgs[cam_idx];
        ftr_new->id_ = new_frame_->id_;
        ftr_new->inv_expo_time_ = state->inv_expo_time;

        // 将此 Feature 添加到该 VisualPoint 的观测列表中
        pt_new->addFrameRef(ftr_new);
        // 复制外部给定的协方差
        pt_new->covariance_ = pt_var.var;
        pt_new->is_normal_initialized_ = true;
        // 对于法向量，保证方向与观察方向一致
        V3D norm_vec = new_frame_->T_f_w_[cam_idx].rotation_matrix() * pt_var.normal;
        V3D dir = new_frame_->T_f_w_[cam_idx] * pt;
        if (dir.dot(norm_vec) < 0)
            pt_new->normal_ = -pt_var.normal;
        else
            pt_new->normal_ = pt_var.normal;
        pt_new->previous_normal_ = pt_new->normal_;

        // 插入该地图点到全局的 voxel map 中
        insertPointIntoVoxelMap(pt_new);
        add_count++;
    }

    ROS_WARN("[ VIO ] Append %d new visual map points (multi-cam)\n", add_count);
}


void VIOManager::updateVisualMapPoints(std::vector<cv::Mat> &imgs)
{
    // 如果没有点，就不做任何更新
    if (total_points == 0)
        return;

    int update_num = 0;

    // 遍历可视化子地图中的所有点
    for (int i = 0; i < total_points; i++)
    {
        VisualPoint* pt = visual_submap->voxel_points[i];
        if (pt == nullptr)
            continue;
        // 如果点已经收敛，就删掉它的非参考 patch（节省内存），并跳过更新
        if (pt->is_converged_)
        {
            pt->deleteNonRefPatchFeatures();
            continue;
        }

        // 取出该点最后一个观测
        // 如果需要基于观测时间做一些间隔判断，也可以在此进行
        if (pt->obs_.empty())
            continue; // 如果没有任何观测，后面也没法比较
        Feature *last_feature = pt->obs_.back();

        // 计算该点在所有相机下的投影
        for (int cam_idx = 0; cam_idx < cams.size(); cam_idx++)
        {
            // 当前相机的位姿
            SE3 pose_cur = new_frame_->T_f_w_[cam_idx];

            // 将点投影到第 cam_idx 个相机
            V2D pc = new_frame_->w2c(pt->pos_, cam_idx);
            if (!cams[cam_idx]->isInFrame(pc.cast<int>(), border))
                continue;  // 不在图像范围内，跳过

            // 拿到“最后一次”引用相机的位姿（或者直接用 last_feature->T_f_w_）
            SE3 pose_ref = last_feature->T_f_w_;

            // Step 1: 判断相机位姿变化（delta_pose）
            SE3 delta_pose = pose_ref * pose_cur.inverse();
            double delta_p = delta_pose.translation().norm();
            // 旋转量
            double cos_val = 0.5 * (delta_pose.rotation_matrix().trace() - 1);
            // acos 要判断边界 [-1,1]
            if (cos_val > 1.0)  cos_val = 1.0;
            if (cos_val < -1.0) cos_val = -1.0;
            double delta_theta = std::acos(cos_val);

            // 根据需求判断是否需要添加观测
            bool add_flag = false;
            if (delta_p > 0.5 || delta_theta > 0.3)
                add_flag = true;

            // Step 2: 判断像素运动距离
            V2D last_px = last_feature->px_;
            double pixel_dist = (pc - last_px).norm();
            if (pixel_dist > 40.0)
                add_flag = true;

            // Step 3: 维持最多 30 条观测
            if (pt->obs_.size() >= 30)
            {
                Feature *ref_ftr = nullptr;
                // 这里用一个示例“找得分最低的观测”或“找与当前位姿最远的观测”，看自己定义
                pt->findMinScoreFeature(new_frame_->pos(cam_idx), ref_ftr);
                if (ref_ftr) pt->deleteFeatureRef(ref_ftr);
            }

            // 如果触发 add_flag，则为该点在此相机下新增一个观测
            if (add_flag)
            {
                // 提取 patch
                float* patch_temp = new float[patch_size_total];
                getImagePatch(imgs[cam_idx], pc, patch_temp, 0);

                // 构造特征
                Vector3d f = cams[cam_idx]->cam2world(pc);
                // 如果在“visual_submap->search_levels[i]”里存了一个金字塔层次，则用它
                int search_level = (visual_submap->search_levels.size() > (size_t)i)
                                   ? visual_submap->search_levels[i] : 0;

                Feature *ftr_new = new Feature(
                        pt,                         // 关联的 VisualPoint
                        patch_temp,                 // 新 patch
                        pc,                         // 投影坐标
                        f,                          // 相机坐标系方向
                        new_frame_->T_f_w_[cam_idx],// 当前相机位姿
                        search_level,
                        cam_idx
                );

                // 补充其他信息
                ftr_new->cam_id_ = cam_idx;
                ftr_new->img_           = imgs[cam_idx];
                ftr_new->id_            = new_frame_->id_;
                ftr_new->inv_expo_time_ = state->inv_expo_time;

                // 将此特征加入该 3D 点
                pt->addFrameRef(ftr_new);

                // 更新计数器
                update_num++;
                update_flag[i] = 1; // 记录一下这个点在该相机下被更新过
            }
        } // end for cam_idx
    } // end for i

    printf("[ VIO ] Update %d points in visual submap (multi-cam)\n", update_num);
}

void VIOManager::updateReferencePatch(const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree*> &plane_map)
{
    if (total_points == 0)
        return;

    // 遍历当前视觉子地图里的所有点
    for (int i = 0; i < (int)visual_submap->voxel_points.size(); i++)
    {
        VisualPoint *pt = visual_submap->voxel_points[i];

        // 筛掉一些不需要处理的点
        if (!pt->is_normal_initialized_ || pt->is_converged_ || pt->obs_.size() <= 5)
            continue;
        if (update_flag[i] == 0)
            continue;

        // ---------------------------------------------------------------------
        // (A) 根据 plane_map 更新点的法向量 normal
        // ---------------------------------------------------------------------
        {
            const V3D &p_w = pt->pos_;
            float loc_xyz[3];
            for (int j = 0; j < 3; j++)
            {
                loc_xyz[j] = p_w[j] / 0.5;   // 这里的体素大小 0.5
                if (loc_xyz[j] < 0)
                    loc_xyz[j] -= 1.0f;
            }
            VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);

            auto it_plane = plane_map.find(position);
            if (it_plane != plane_map.end())
            {
                VoxelOctoTree *current_octo = it_plane->second->find_correspond(p_w);
                if (current_octo->plane_ptr_->is_plane_)
                {
                    VoxelPlane &plane = *current_octo->plane_ptr_;
                    float dis_to_plane =
                            plane.normal_.dot(p_w) + plane.d_;
                    float dis_to_plane_abs = std::fabs(dis_to_plane);

                    float dis_to_center = (plane.center_ - p_w).squaredNorm();
                    float range_dis = std::sqrt(dis_to_center - dis_to_plane*dis_to_plane);

                    if (range_dis <= 3.0f * plane.radius_)
                    {
                        // sigma_l
                        Eigen::Matrix<double, 1, 6> J_nq;
                        J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
                        J_nq.block<1, 3>(0, 3) = -plane.normal_.transpose();
                        double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
                        sigma_l += plane.normal_.transpose() * pt->covariance_ * plane.normal_;

                        if (dis_to_plane_abs < 3.0 * std::sqrt(sigma_l))
                        {
                            // 根据 previous_normal_ 和 plane.normal_ 保证方向一致
                            if (pt->previous_normal_.dot(plane.normal_) < 0)
                                pt->normal_ = -plane.normal_;
                            else
                                pt->normal_ = plane.normal_;

                            double normal_update = (pt->normal_ - pt->previous_normal_).norm();
                            pt->previous_normal_ = pt->normal_;

                            // 若法向量变化很小且观测>=10，认为点已收敛
                            if (normal_update < 0.0001 && pt->obs_.size() > 10)
                            {
                                pt->is_converged_ = true;
                            }
                        }
                    }
                }
            }
        }

        // ---------------------------------------------------------------------
        // (B) 在所有观测 obs_ 中找“score”最高的 patch，更新 pt->ref_patch
        // ---------------------------------------------------------------------
        float score_max = -1e6;
        Feature *best_ref_ftr = nullptr;

        // 遍历每个观测
        for (auto it = pt->obs_.begin(); it != pt->obs_.end(); ++it)
        {
            Feature *ref_patch_temp = *it;
            if (!ref_patch_temp)
                continue;

            // patch_temp 指向该 feature 的图像 patch
            float *patch_temp = ref_patch_temp->patch_;

            // 计算该观测下的 cos_angle
            // ref_patch_temp->T_f_w_ 是相机位姿(世界->相机)
            // 那么将 pt->pos_ 转到相机坐标系
            // pf = Rcw * p_w + Pcw
            V3D pf = ref_patch_temp->T_f_w_ * pt->pos_;  // world->cam


            // 将法向量转到相机坐标系
            V3D norm_vec = ref_patch_temp->T_f_w_.rotation_matrix() * pt->normal_;

            // pf 归一化，然后与 norm_vec 做点积
            pf.normalize();
            double cos_angle = pf.dot(norm_vec);

            // 如果想滤除过小 cos_angle，如 < 0.86 (大约30度)
            // if(std::fabs(cos_angle) < 0.86)
            //     continue;

            // 若还没计算 mean_，则计算一次
            if (std::fabs(ref_patch_temp->mean_) < 1e-6)
            {
                float sum_val = std::accumulate(patch_temp, patch_temp + patch_size_total, 0.0f);
                float mean_val = sum_val / (float)patch_size_total;
                ref_patch_temp->mean_ = mean_val;
            }
            float ref_mean = ref_patch_temp->mean_;

            // --------------------
            // 计算此 patch 与所有其他观测 patch 的 NCC 平均值
            // --------------------
            float sumNCC = 0.0f;
            int countNCC = 0;

            for (auto itm = pt->obs_.begin(); itm != pt->obs_.end(); ++itm)
            {
                if((*itm)->id_ == ref_patch_temp->id_)
                    continue;

                // 另一个 patch
                float *patch_cache = (*itm)->patch_;
                // 若还没算 mean
                if (std::fabs((*itm)->mean_) < 1e-6)
                {
                    float sum_val2 = std::accumulate(patch_cache, patch_cache + patch_size_total, 0.0f);
                    (*itm)->mean_ = sum_val2 / (float)patch_size_total;
                }
                float other_mean = (*itm)->mean_;

                // 计算与 patch_temp 的 NCC
                double numerator   = 0.0;
                double denominator1= 0.0;
                double denominator2= 0.0;
                for (int ind = 0; ind < patch_size_total; ind++)
                {
                    double diff1 = (double)(patch_temp[ind] - ref_mean);
                    double diff2 = (double)(patch_cache[ind] - other_mean);
                    numerator    += diff1 * diff2;
                    denominator1 += diff1 * diff1;
                    denominator2 += diff2 * diff2;
                }
                double ncc_val = numerator / std::sqrt(denominator1*denominator2 + 1e-10);
                // 这里加绝对值
                sumNCC += std::fabs(ncc_val);
                countNCC++;
            }

            // 取多观测 NCC 的均值
            float NCC_avg = (countNCC>0) ? (sumNCC / (float)countNCC) : 0.0f;

            // 综合得分：score = NCC + cos_angle
            float score = NCC_avg + (float)cos_angle;
            ref_patch_temp->score_ = score;

            // 如果分数最高，则更新
            if (score > score_max)
            {
                score_max     = score;
                best_ref_ftr  = ref_patch_temp;
            }
        } // end for obs_

        // 最终，把得分最高的 feature 设为 ref_patch
        if (best_ref_ftr)
        {
            pt->ref_patch = best_ref_ftr;
            pt->has_ref_patch_ = true;
        }
    } // end for i
}
//仅对cam0
void VIOManager::projectPatchFromRefToCur(const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
    if (total_points == 0) return;
    // if(new_frame_->id_ != 2) return; //124

    int patch_size = 25;
    string dir = string(ROOT_DIR) + "Log/ref_cur_combine/";

    cv::Mat result = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat result_normal = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat result_dense = cv::Mat::zeros(height, width, CV_8UC1);

    cv::Mat img_photometric_error = new_frame_->imgs_[0].clone();

    uchar *it = (uchar *)result.data;
    uchar *it_normal = (uchar *)result_normal.data;
    uchar *it_dense = (uchar *)result_dense.data;

    struct pixel_member
    {
        Vector2f pixel_pos;
        uint8_t pixel_value;
    };

    int num = 0;
    for (int i = 0; i < visual_submap->voxel_points.size(); i++)
    {
        VisualPoint *pt = visual_submap->voxel_points[i];

        if (pt->is_normal_initialized_)
        {
            Feature *ref_ftr;
            ref_ftr = pt->ref_patch;

            // Feature* ref_ftr;
            V2D pc(new_frame_->w2c(pt->pos_,0));
            V2D pc_prior(new_frame_->w2c_prior(pt->pos_,0));

            V3D norm_vec(ref_ftr->T_f_w_.rotation_matrix() * pt->normal_);
            V3D pf(ref_ftr->T_f_w_ * pt->pos_);

            if (pf.dot(norm_vec) < 0) norm_vec = -norm_vec;

            // norm_vec << norm_vec(1), norm_vec(0), norm_vec(2);
            cv::Mat img_cur = new_frame_->imgs_[0];
            cv::Mat img_ref = ref_ftr->img_;

            SE3 T_cur_ref = new_frame_->T_f_w_[0] * ref_ftr->T_f_w_.inverse();
            Matrix2d A_cur_ref;
            getWarpMatrixAffineHomography(*cams[0], ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref);

            // const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
            int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);

            double D = A_cur_ref.determinant();
            if (D > 3) continue;

            num++;

            cv::Mat ref_cur_combine_temp;
            int radius = 20;
            cv::hconcat(img_cur, img_ref, ref_cur_combine_temp);
            cv::cvtColor(ref_cur_combine_temp, ref_cur_combine_temp, CV_GRAY2BGR);

            getImagePatch(img_cur, pc, patch_buffer.data(), 0);

            float error_est = 0.0;
            float error_gt = 0.0;

            for (int ind = 0; ind < patch_size_total; ind++)
            {
                error_est += (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]) *
                             (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]);
            }
            std::string ref_est = "ref_est " + std::to_string(1.0 / ref_ftr->inv_expo_time_);
            std::string cur_est = "cur_est " + std::to_string(1.0 / state->inv_expo_time);
            std::string cur_propa = "cur_gt " + std::to_string(error_gt);
            std::string cur_optimize = "cur_est " + std::to_string(error_est);

            cv::putText(ref_cur_combine_temp, ref_est, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - 40, ref_ftr->px_[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4,
                        cv::Scalar(0, 255, 0), 1, 8, 0);

            cv::putText(ref_cur_combine_temp, cur_est, cv::Point2f(pc[0] - 40, pc[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8, 0);
            cv::putText(ref_cur_combine_temp, cur_propa, cv::Point2f(pc[0] - 40, pc[1] + 60), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 255), 1, 8,
                        0);
            cv::putText(ref_cur_combine_temp, cur_optimize, cv::Point2f(pc[0] - 40, pc[1] + 80), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8,
                        0);

            cv::rectangle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - radius, ref_ftr->px_[1] - radius),
                          cv::Point2f(ref_ftr->px_[0] + img_cur.cols + radius, ref_ftr->px_[1] + radius), cv::Scalar(0, 0, 255), 1);
            cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc[0] - radius, pc[1] - radius), cv::Point2f(pc[0] + radius, pc[1] + radius),
                          cv::Scalar(0, 255, 0), 1);
            cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc_prior[0] - radius, pc_prior[1] - radius),
                          cv::Point2f(pc_prior[0] + radius, pc_prior[1] + radius), cv::Scalar(255, 255, 255), 1);
            cv::circle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols, ref_ftr->px_[1]), 1, cv::Scalar(0, 0, 255), -1, 8);
            cv::circle(ref_cur_combine_temp, cv::Point2f(pc[0], pc[1]), 1, cv::Scalar(0, 255, 0), -1, 8);
            cv::circle(ref_cur_combine_temp, cv::Point2f(pc_prior[0], pc_prior[1]), 1, cv::Scalar(255, 255, 255), -1, 8);
            cv::imwrite(dir + std::to_string(new_frame_->id_) + "_" + std::to_string(ref_ftr->id_) + "_" + std::to_string(num) + ".png",
                        ref_cur_combine_temp);

            std::vector<std::vector<pixel_member>> pixel_warp_matrix;

            for (int y = 0; y < patch_size; ++y)
            {
                vector<pixel_member> pixel_warp_vec;
                for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
                {
                    Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
                    px_patch *= (1 << search_level);
                    const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
                    uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

                    const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
                    if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
                        continue;
                    else
                    {
                        pixel_member pixel_warp;
                        pixel_warp.pixel_pos << px[0], px[1];
                        pixel_warp.pixel_value = pixel_value;
                        pixel_warp_vec.push_back(pixel_warp);
                    }
                }
                pixel_warp_matrix.push_back(pixel_warp_vec);
            }

            float x_min = 1000;
            float y_min = 1000;
            float x_max = 0;
            float y_max = 0;

            for (int i = 0; i < pixel_warp_matrix.size(); i++)
            {
                vector<pixel_member> pixel_warp_row = pixel_warp_matrix[i];
                for (int j = 0; j < pixel_warp_row.size(); j++)
                {
                    float x_temp = pixel_warp_row[j].pixel_pos[0];
                    float y_temp = pixel_warp_row[j].pixel_pos[1];
                    if (x_temp < x_min) x_min = x_temp;
                    if (y_temp < y_min) y_min = y_temp;
                    if (x_temp > x_max) x_max = x_temp;
                    if (y_temp > y_max) y_max = y_temp;
                }
            }
            int x_min_i = floor(x_min);
            int y_min_i = floor(y_min);
            int x_max_i = ceil(x_max);
            int y_max_i = ceil(y_max);
            Matrix2f A_cur_ref_Inv = A_cur_ref.inverse().cast<float>();
            for (int i = x_min_i; i < x_max_i; i++)
            {
                for (int j = y_min_i; j < y_max_i; j++)
                {
                    Eigen::Vector2f pc_temp(i, j);
                    Vector2f px_patch = A_cur_ref_Inv * (pc_temp - pc.cast<float>());
                    if (px_patch[0] > (-patch_size / 2 * (1 << search_level)) && px_patch[0] < (patch_size / 2 * (1 << search_level)) &&
                        px_patch[1] > (-patch_size / 2 * (1 << search_level)) && px_patch[1] < (patch_size / 2 * (1 << search_level)))
                    {
                        const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
                        uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);
                        it_normal[width * j + i] = pixel_value;
                    }
                }
            }
        }
    }
    for (int i = 0; i < visual_submap->voxel_points.size(); i++)
    {
        VisualPoint *pt = visual_submap->voxel_points[i];

        if (!pt->is_normal_initialized_) continue;

        Feature *ref_ftr;
        V2D pc(new_frame_->w2c(pt->pos_,0));
        ref_ftr = pt->ref_patch;

        Matrix2d A_cur_ref;
        getWarpMatrixAffine(*cams[0], ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ [0]* ref_ftr->T_f_w_.inverse(), 0, 0,
                            patch_size_half, A_cur_ref);
        int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);
        double D = A_cur_ref.determinant();
        if (D > 3) continue;

        cv::Mat img_cur = new_frame_->imgs_[0];
        cv::Mat img_ref = ref_ftr->img_;
        for (int y = 0; y < patch_size; ++y)
        {
            for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
            {
                Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
                px_patch *= (1 << search_level);
                const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
                uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

                const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
                if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
                    continue;
                else
                {
                    int col = int(px[0]);
                    int row = int(px[1]);
                    it[width * row + col] = pixel_value;
                }
            }
        }
    }
    cv::Mat ref_cur_combine;
    cv::Mat ref_cur_combine_normal;
    cv::Mat ref_cur_combine_error;

    cv::hconcat(result, new_frame_->imgs_[0], ref_cur_combine);
    cv::hconcat(result_normal, new_frame_->imgs_[0], ref_cur_combine_normal);

    cv::cvtColor(ref_cur_combine, ref_cur_combine, CV_GRAY2BGR);
    cv::cvtColor(ref_cur_combine_normal, ref_cur_combine_normal, CV_GRAY2BGR);
    cv::absdiff(img_photometric_error, result_normal, img_photometric_error);
    cv::hconcat(img_photometric_error, new_frame_->imgs_[0], ref_cur_combine_error);

    cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + ".png", ref_cur_combine);
    cv::imwrite(dir + std::to_string(new_frame_->id_) + +"_0_" +
                "photometric"
                ".png",
                ref_cur_combine_error);
    cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + "normal" + ".png", ref_cur_combine_normal);
}

void VIOManager::precomputeReferencePatches(int level)
{

    // 如果 total_points 为0，直接退出
    double t1 = omp_get_wtime();
    if (total_points == 0) return;

    const int num_cams = (int)cams.size();
    const int H_DIM = total_points * patch_size_total * num_cams;

    H_sub_inv.resize(H_DIM, 6);
    H_sub_inv.setZero();

    int row_offset_cam = 0;

    for (int cam_idx = 0; cam_idx < num_cams; cam_idx++)
    {

        double fx_i = cams[cam_idx]->fx();
        double fy_i = cams[cam_idx]->fy();

        for (int i = 0; i < total_points; i++)
        {
            VisualPoint *pt = visual_submap->voxel_points[i];
            if (!pt) {
                ROS_WARN(" pt for Feature reference at point %d is empty, skipping.", i);
                continue;
            }

            Feature *ref_ftr = pt->ref_patch;
            if (!ref_ftr) {
                ROS_WARN(" ref_ftr for Feature reference at point %d is empty, skipping.", i);
                continue;
            }

            cv::Mat &img = ref_ftr->img_;
            if (img.empty()) {
                ROS_WARN(" Image for Feature reference at point %d is empty, skipping.", i);
                continue;
            }


            double depth = (pt->pos_ - ref_ftr->pos()).norm();
            V3D pf = ref_ftr->f_ * depth;

            V2D pc = ref_ftr->px_;
            M3D R_ref_w = ref_ftr->T_f_w_.rotation_matrix();

            MD(2, 3) Jdpi;
            computeProjectionJacobian(cam_idx, pf, Jdpi);

            M3D p_hat;
            p_hat << SKEW_SYM_MATRX(pt->pos_);

            int search_level   = visual_submap->search_levels[i];
            int pyramid_level  = level + search_level;
            int scale          = (1 << pyramid_level);
            float inv_scale    = 1.0f / scale;

            float u_ref = pc[0];
            float v_ref = pc[1];
            int u_ref_i = (int)std::floor(u_ref / scale) * scale;
            int v_ref_i = (int)std::floor(v_ref / scale) * scale;
            float subpix_u_ref = (u_ref - u_ref_i) / scale;
            float subpix_v_ref = (v_ref - v_ref_i) / scale;
            float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
            float w_ref_tr =          subpix_u_ref  * (1.0f - subpix_v_ref);
            float w_ref_bl = (1.0f - subpix_u_ref) *          subpix_v_ref;
            float w_ref_br =          subpix_u_ref  *          subpix_v_ref;

            int row_offset_pt = row_offset_cam + i * patch_size_total;

            for (int px_y = 0; px_y < patch_size; px_y++)
            {
                int row_img = v_ref_i + px_y * scale - patch_size_half * scale;
                if (row_img < 1 || row_img >= img.rows - 1)
                {
                    continue;
                }

                int col_start = (u_ref_i - patch_size_half * scale);
                if (col_start < 1 || col_start >= img.cols - 1)
                {
                    continue;
                }

                uint8_t* img_ptr = (uint8_t*)img.data + row_img * img.cols + col_start;

                for (int px_x = 0; px_x < patch_size; px_x++, img_ptr += scale)
                {
                    if ((col_start + px_x * scale) < 1 || (col_start + px_x * scale) >= (img.cols - 1))
                    {
                        continue;
                    }

                    float du = 0.5f * (
                            (w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2]
                             + w_ref_bl * img_ptr[scale * img.cols + scale]
                             + w_ref_br * img_ptr[scale * img.cols + scale * 2])
                            - (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0]
                               + w_ref_bl * img_ptr[scale * img.cols - scale]
                               + w_ref_br * img_ptr[scale * img.cols])
                    );
                    float dv = 0.5f * (
                            (w_ref_tl * img_ptr[scale * img.cols] + w_ref_tr * img_ptr[scale + scale * img.cols]
                             + w_ref_bl * img_ptr[scale * img.cols * 2] + w_ref_br * img_ptr[scale * img.cols * 2 + scale])
                            - (w_ref_tl * img_ptr[-scale * img.cols] + w_ref_tr * img_ptr[-scale * img.cols + scale]
                               + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale])
                    );

                    MD(1,2) Jimg;
                    Jimg << du, dv;
                    Jimg *= inv_scale;

                    MD(1,3) J_dphi = Jimg * Jdpi * (R_ref_w * p_hat);
                    MD(1,3) J_dp   = -Jimg * Jdpi * R_ref_w;

                    MD(1,3) JdR = J_dphi * Jdphi_dR_vec[cam_idx] + J_dp * Jdp_dR_vec[cam_idx];
                    MD(1,3) Jdt = J_dp * Jdp_dt_vec[cam_idx];

                    int row_cur = row_offset_pt + px_y * patch_size + px_x;
                    if (row_cur >= 0 && row_cur < H_DIM)
                    {
                        H_sub_inv.block<1,6>(row_cur, 0) << JdR, Jdt;
                    }
                    else
                    {
                        ROS_WARN("[precomputeReferencePatches] row_cur out of bounds: %d", row_cur);
                    }
                }
            }
        }

        row_offset_cam += total_points * patch_size_total;
    }

    has_ref_patch_cache = true;
    compute_jacobian_time += (omp_get_wtime() - t1);
}

void VIOManager::updateStateInverse(const std::vector<cv::Mat>& imgs, int level)
{
    if (total_points == 0) return;
    // 保存旧状态（用于回退）
    StatesGroup old_state = (*state);

    // 多相机情况下，全局残差和雅可比的测量维度为每个相机的数据堆叠
    const int num_cams = static_cast<int>(imgs.size());
    const int H_DIM = total_points * patch_size_total * num_cams;
    VectorXd z(H_DIM);
    z.setZero();
    MatrixXd H_sub(H_DIM, 6); // 6列：旋转3 + 平移3
    H_sub.setZero();

    bool EKF_end = false;
    float last_error = std::numeric_limits<float>::max();

    compute_jacobian_time = 0.0;
    update_ekf_time = 0.0;
    // 用于平移部分补偿（取当前状态平移的反对称矩阵）
    M3D P_wi_hat;
    P_wi_hat << SKEW_SYM_MATRX(state->pos_end);

    // 如果参考 patch 尚未预计算，则先预计算
    if (!has_ref_patch_cache) {
        ROS_INFO("[updateStateInverse] Precomputing reference patches...");
        precomputeReferencePatches(level);
    }

    // 主迭代循环
    for (int iter = 0; iter < max_iterations; iter++)
    {
        double t1 = omp_get_wtime();
        int n_meas = 0;
        float error = 0.0f;

        // 当前机体状态（各相机共享同一状态）
        M3D Rwi(state->rot_end);
        V3D Pwi(state->pos_end);

        // row_offset_cam 用于区分不同相机在全局数据中的起始行
        int row_offset_cam = 0;

        // 遍历所有相机
        for (int cam_idx = 0; cam_idx < num_cams; cam_idx++)
        {
            if (imgs[cam_idx].empty()) {
                ROS_WARN_STREAM("[updateStateInverse] Cam[" << cam_idx << "] image is empty, skipping.");
                row_offset_cam += (total_points * patch_size_total);
                continue;
            }

            // 使用当前相机的外参计算当前帧的投影变换
            // 例如：Rcw_vec[cam_idx] = Rci_vec[cam_idx] * Rwi.transpose();
            //          Pcw_vec[cam_idx] = -Rci_vec[cam_idx] * Rwi.transpose() * Pwi + Pci_vec[cam_idx];
            Rcw_vec[cam_idx] = Rci_vec[cam_idx] * Rwi.transpose();
            Pcw_vec[cam_idx] = -Rci_vec[cam_idx] * Rwi.transpose() * Pwi + Pci_vec[cam_idx];
            Jdp_dt_vec[cam_idx] = Rci_vec[cam_idx] * Rwi.transpose();

            const cv::Mat &img_cur = imgs[cam_idx];

#ifdef MP_EN
            omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for reduction(+:error, n_meas)
#endif
            for (int i_pt = 0; i_pt < total_points; i_pt++)
            {
                float patch_error = 0.0f;
                const int scale = (1 << level);

                VisualPoint *pt = visual_submap->voxel_points[i_pt];
                if (!pt) continue;

                // 将 3D 点投影到当前相机
                V3D pf = Rcw_vec[cam_idx] * pt->pos_ + Pcw_vec[cam_idx];
                if (pf.z() < 1e-6) continue;
                V2D pc = cams[cam_idx]->world2cam(pf);

                // 计算亚像素坐标及插值权重
                int u_ref_i = static_cast<int>(std::floor(pc[0] / scale)) * scale;
                int v_ref_i = static_cast<int>(std::floor(pc[1] / scale)) * scale;
                float subpix_u_ref = (pc[0] - u_ref_i) / scale;
                float subpix_v_ref = (pc[1] - v_ref_i) / scale;
                float w_ref_tl = (1.f - subpix_u_ref) * (1.f - subpix_v_ref);
                float w_ref_tr =         subpix_u_ref  * (1.f - subpix_v_ref);
                float w_ref_bl = (1.f - subpix_u_ref) *        subpix_v_ref;
                float w_ref_br =         subpix_u_ref  *       subpix_v_ref;

                // 参考 patch 数据（光度值）与参考曝光参数
                const std::vector<float> &P_patch = visual_submap->warp_patch[i_pt];
                double inv_ref_expo = visual_submap->inv_expo_list[i_pt];

                // 当前点在当前相机的测量在全局数据中的起始行
                int row_offset_pt = row_offset_cam + i_pt * patch_size_total;

                // 遍历 patch 内部：采用外层循环按行（px_y），内层按列（px_x）
                for (int px_y = 0; px_y < patch_size; px_y++)
                {
                    int row_img = v_ref_i + px_y * scale - patch_size_half * scale;
                    if (row_img < 1 || row_img >= (img_cur.rows - 1))
                        continue;
                    int col_start = u_ref_i - patch_size_half * scale;
                    if (col_start < 1 || col_start >= (img_cur.cols - 1))
                        continue;
                    uint8_t *img_ptr = (uint8_t*)img_cur.data + row_img * img_cur.cols + col_start;

                    for (int px_x = 0; px_x < patch_size; px_x++, img_ptr += scale)
                    {
                        int cur_col = col_start + px_x * scale;
                        if (cur_col < 1 || cur_col >= (img_cur.cols - 1))
                            continue;

                        // 计算图像梯度
                        float du = 0.5f * (
                                (w_ref_tl * img_ptr[scale] +
                                 w_ref_tr * img_ptr[scale * 2] +
                                 w_ref_bl * img_ptr[scale * img_cur.cols + scale] +
                                 w_ref_br * img_ptr[scale * img_cur.cols + scale * 2])
                                -
                                (w_ref_tl * img_ptr[-scale] +
                                 w_ref_tr * img_ptr[0] +
                                 w_ref_bl * img_ptr[scale * img_cur.cols - scale] +
                                 w_ref_br * img_ptr[scale * img_cur.cols])
                        );
                        float dv = 0.5f * (
                                (w_ref_tl * img_ptr[scale * img_cur.cols] +
                                 w_ref_tr * img_ptr[scale + scale * img_cur.cols] +
                                 w_ref_bl * img_ptr[scale * img_cur.cols * 2] +
                                 w_ref_br * img_ptr[scale * img_cur.cols * 2 + scale])
                                -
                                (w_ref_tl * img_ptr[-scale * img_cur.cols] +
                                 w_ref_tr * img_ptr[-scale * img_cur.cols + scale] +
                                 w_ref_bl * img_ptr[0] +
                                 w_ref_br * img_ptr[scale])
                        );

                        MD(1,2) Jimg;
                        Jimg << du, dv;
                        Jimg = Jimg * state->inv_expo_time;
                        Jimg = Jimg * (1.0f / scale);

                        // 计算投影雅可比：注意使用当前 cam_idx
                        MD(2,3) Jdpi;
                        computeProjectionJacobian(cam_idx, pf, Jdpi);
                        // p_hat 为 pf 的反对称矩阵
                        M3D p_hat;
                        p_hat << SKEW_SYM_MATRX(pf);
                        MD(1,3) J_dphi = Jimg * Jdpi * p_hat;
                        MD(1,3) J_dp   = -Jimg * Jdpi;
                        MD(1,3) JdR_local = J_dphi * Jdphi_dR_vec[cam_idx] + J_dp * Jdp_dR_vec[cam_idx];
                        MD(1,3) Jdt_local = J_dp * Jdp_dt_vec[cam_idx];

                        // 根据预计算参考 patch，取对应雅可比
                        int row_idx = row_offset_pt + px_y * patch_size + px_x;
                        if (row_idx < 0 || row_idx >= H_DIM) continue;
                        MD(1,3) J_dR_ref = H_sub_inv.block<1,3>(row_idx, 0);
                        MD(1,3) J_dt_ref = H_sub_inv.block<1,3>(row_idx, 3);
                        // 根据当前状态对参考雅可比进行补偿
                        MD(1,3) JdR_final = J_dR_ref * Rwi + J_dt_ref * P_wi_hat * Rwi;
                        MD(1,3) Jdt_final = J_dt_ref * Rwi;

                        // 写入当前帧的雅可比矩阵
                        H_sub.block<1,6>(row_idx, 0) << JdR_final, Jdt_final;

                        // 插值获得当前像素的灰度值
                        float cur_val = w_ref_tl * img_ptr[0] +
                                        w_ref_tr * img_ptr[scale] +
                                        w_ref_bl * img_ptr[scale * img_cur.cols] +
                                        w_ref_br * img_ptr[scale * img_cur.cols + scale];

                        int patch_idx = px_y * patch_size + px_x;
                        double res = state->inv_expo_time * cur_val - inv_ref_expo * P_patch[patch_size_total * level + patch_idx];
                        z(row_idx) = res;
                        patch_error += res * res;
                        n_meas++;

                        // 可选调试：打印部分残差值
                        // ROS_INFO_STREAM("[updateStateInverse] Cam[" << cam_idx << "], pt[" << i_pt << "], pixel(" << px_x << "," << px_y << ") res: " << res);
                    } // end for px_x
                } // end for px_y

                visual_submap->errors[i_pt] = patch_error;
#pragma omp atomic
                error += patch_error;
            } // end for each 3D point in current camera

            // 更新行偏移，为下一相机预留空间
            row_offset_cam += (total_points * patch_size_total);
        } // end for cam_idx

        if (n_meas > 0)
            error /= n_meas;
        compute_jacobian_time += omp_get_wtime() - t1;

        // ---------- EKF / Gauss-Newton 状态更新 ----------
        double t3 = omp_get_wtime();
        if (error <= last_error && error > 0)  // 确保 error 正常
        {
            old_state = (*state);
            last_error = error;

            auto H_sub_T = H_sub.transpose();
            H_T_H.setZero();
            G.setZero();
            H_T_H.block<6,6>(0,0) = H_sub_T * H_sub;
            double det = H_T_H.determinant();

            MD(DIM_STATE, DIM_STATE) K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
            auto HTz = H_sub_T * z;
            auto vec = (*state_propagat) - (*state);
            G.block<DIM_STATE,6>(0,0) = K_1.block<DIM_STATE,6>(0,0) * H_T_H.block<6,6>(0,0);
            MD(DIM_STATE,1) solution = -K_1.block<DIM_STATE,6>(0,0) * HTz + vec - G.block<DIM_STATE,6>(0,0) * vec.block<6,1>(0,0);

            (*state) += solution;
            auto rot_add = solution.block<3,1>(0,0);
            auto t_add   = solution.block<3,1>(3,0);
            if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))
                EKF_end = true;
        }
        else
        {
            (*state) = old_state;
            EKF_end = true;
        }
        update_ekf_time += omp_get_wtime() - t3;

        if (iter == max_iterations - 1 || EKF_end)
            break;
    } // end for iteration

    ROS_INFO_STREAM("[updateStateInverse] Finished iterations, final error: " << last_error);
}

inline float huberWeight(float r, float huber_delta = 5.0f)
{
    float abs_r = fabs(r);
    // 若在阈值范围内，权重=1，否则 = δ / |r|
    if (abs_r < huber_delta) {
        return 1.0f;
    } else {
        return huber_delta / abs_r;
    }
}

void VIOManager::updateState(const std::vector<cv::Mat> &imgs, int level)
{
    // 如果没有点，则直接退出
    if (total_points == 0) {
        ROS_WARN("[updateState] total_points is 0, exiting.");
        return;
    }

    // 保存旧状态，用于在迭代中回退
    StatesGroup old_state = (*state);

    // 多相机情况下的测量维度
    const int num_cams = (int)cams.size();
    // 每个点对应 patch_size_total 像素，每个相机都对这些像素形成约束
    const int H_DIM = total_points * patch_size_total * num_cams;

    // 残差向量 z，雅可比矩阵 H_sub(H_DIM x 7)
    VectorXd z(H_DIM);
    z.setZero();
    MatrixXd H_sub(H_DIM, 7);
    H_sub.setZero();

    bool EKF_end = false;
    float last_error = std::numeric_limits<float>::max();

    for (int iteration = 0; iteration < max_iterations; iteration++)
    {
        double t1 = omp_get_wtime();

        // 当前滤波/优化状态下的 Rwi, Pwi, 以及 inv_expo_time
        M3D Rwi(state->rot_end);
        V3D Pwi(state->pos_end);
        double cur_inv_expo = state->inv_expo_time;

        // 误差统计
        float error = 0.0f;
        int n_meas = 0;

        // row_offset_cam 用于计算在大残差向量/雅可比矩阵中的起始行
        int row_offset_cam = 0;

        // ========== 遍历所有相机 ==========
        for (int cam_idx = 0; cam_idx < num_cams; cam_idx++)
        {
            if (imgs[cam_idx].empty()) {
                ROS_WARN("[updateState] skip empty image for cam[%d]", cam_idx);
                // 跳过，但要移动行偏移
                row_offset_cam += (total_points * patch_size_total);
                continue;
            }

            // 计算相机在当前状态下的位姿: Rcw, Pcw
            Rcw_vec[cam_idx] = Rci_vec[cam_idx] * Rwi.transpose();
            Pcw_vec[cam_idx] = -Rci_vec[cam_idx] * Rwi.transpose() * Pwi + Pci_vec[cam_idx];

            // 用于平移雅可比的外参变换
            Jdp_dt_vec[cam_idx] = Rci_vec[cam_idx] * Rwi.transpose();

            const cv::Mat &img_cur = imgs[cam_idx];

#ifdef MP_EN
            omp_set_num_threads(MP_PROC_NUM);
#endif
#pragma omp parallel for reduction(+:error,n_meas)
            for (int i_pt = 0; i_pt < total_points; i_pt++)
            {
                // 临时变量
                float patch_error = 0.0f;

                VisualPoint *pt = visual_submap->voxel_points[i_pt];
                if (!pt) continue;

                // 将 3D 点投影到该相机
                V3D pf = Rcw_vec[cam_idx] * pt->pos_ + Pcw_vec[cam_idx];
                if (pf.z() < 1e-6)
                    continue;

                V2D pc = cams[cam_idx]->world2cam(pf);

                // 计算 2D 投影雅可比(2x3)
                MD(2,3) Jdpi;
                computeProjectionJacobian(cam_idx, pf, Jdpi);

                // pf 的反对称矩阵
                M3D p_hat;
                p_hat << SKEW_SYM_MATRX(pf);

                // 尺度 level
                int scale = (1 << level);
                float inv_scale = 1.0f / scale;

                float u_ref = pc[0];
                float v_ref = pc[1];
                int u_ref_i = floorf(u_ref / scale) * scale;
                int v_ref_i = floorf(v_ref / scale) * scale;
                float subpix_u_ref = (u_ref - u_ref_i) * inv_scale;
                float subpix_v_ref = (v_ref - v_ref_i) * inv_scale;
                float w_ref_tl = (1.f - subpix_u_ref) * (1.f - subpix_v_ref);
                float w_ref_tr = subpix_u_ref * (1.f - subpix_v_ref);
                float w_ref_bl = (1.f - subpix_u_ref) * subpix_v_ref;
                float w_ref_br = subpix_u_ref * subpix_v_ref;

                // warp_patch[i_pt] 是参考 patch；inv_expo_list[i_pt] 是参考曝光倒数
                const std::vector<float> &P = visual_submap->warp_patch[i_pt];
                double inv_ref_expo = visual_submap->inv_expo_list[i_pt];

                // 如果 patch 数据不够(无效)，跳过
                if ((int)P.size() <= patch_size_total * level)
                    continue;

                // row_offset_pt 是该相机下、该点在大残差向量中的起始行
                int row_offset_pt = row_offset_cam + i_pt * patch_size_total;

                // 遍历 patch
                for (int px_x = 0; px_x < patch_size; px_x++)
                {
                    int row_img = v_ref_i + px_x * scale - patch_size_half * scale;
                    if (row_img < 1 || row_img >= (img_cur.rows - 1))
                        continue;

                    // 起始指针
                    uint8_t *img_ptr = (uint8_t *)img_cur.data + row_img * width + (u_ref_i - patch_size_half * scale);

                    for (int px_y = 0; px_y < patch_size; px_y++, img_ptr += scale)
                    {
                        int col_img = (u_ref_i - patch_size_half * scale) + px_y * scale;
                        if (col_img < 1 || col_img >= (img_cur.cols - 1))
                            continue;

                        // 计算图像梯度 du,dv (以子像素插值方式)
                        float du = 0.5f * (
                                (w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2]
                                 + w_ref_bl * img_ptr[scale * width + scale]
                                 + w_ref_br * img_ptr[scale * width + scale * 2])
                                - (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0]
                                   + w_ref_bl * img_ptr[scale * width - scale]
                                   + w_ref_br * img_ptr[scale * width])
                        );
                        float dv = 0.5f * (
                                (w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width]
                                 + w_ref_bl * img_ptr[scale * width * 2] + w_ref_br * img_ptr[scale * width * 2 + scale])
                                - (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale]
                                   + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale])
                        );

                        // 光度雅可比(1x2)
                        MD(1,2) Jimg;
                        Jimg << du, dv;
                        // 对当前帧曝光因子的影响
                        Jimg *= (float)cur_inv_expo;
                        Jimg *= inv_scale;

                        // 求对旋转 & 平移的导数
                        // Jdphi = Jimg * Jdpi * p_hat
                        MD(1,3) Jdphi = Jimg * Jdpi * p_hat;
                        // Jdp   = - Jimg * Jdpi
                        MD(1,3) Jdp   = -Jimg * Jdpi;
                        MD(1,3) JdR   = Jdphi * Jdphi_dR_vec[cam_idx]
                                        + Jdp   * Jdp_dR_vec[cam_idx];
                        MD(1,3) Jdt   = Jdp * Jdp_dt_vec[cam_idx];

                        // 计算当前像素的灰度值
                        float cur_val = w_ref_tl * img_ptr[0]
                                        + w_ref_tr * img_ptr[scale]
                                        + w_ref_bl * img_ptr[scale * width]
                                        + w_ref_br * img_ptr[scale * width + scale];

                        // 参考 patch 中对应像素
                        int idx_patch = px_x * patch_size + px_y;
                        float ref_val = P[patch_size_total * level + idx_patch];

                        // 残差: res = (inv_expo_time * cur_val) - (inv_ref_expo * ref_val)
                        float res = (float)(cur_inv_expo * cur_val - inv_ref_expo * ref_val);

                        // -------------- 计算对 inv_expo_time 的导数(第7列) --------------
                        // res = inv_expo_time * cur_val - inv_ref_expo * ref_val
                        // => d(res)/d(inv_expo_time) = cur_val
                        float J_expo = cur_val;

                        // 使用 Huber 核
                        float w = huberWeight(res, 5.f);   // huber_delta = 5.0f 可改
                        float sqrt_w = sqrtf(w);

                        // 写入残差向量(乘 sqrt_w)
                        int row_here = row_offset_pt + idx_patch;
                        z(row_here) = sqrt_w * res;

                        // 将对应行的雅可比也乘以 sqrt_w
                        // 第0..2列 => 旋转, 第3..5列 => 平移, 第6列 => inv_expo_time
                        H_sub(row_here, 0) = sqrt_w * JdR(0);
                        H_sub(row_here, 1) = sqrt_w * JdR(1);
                        H_sub(row_here, 2) = sqrt_w * JdR(2);

                        H_sub(row_here, 3) = sqrt_w * Jdt(0);
                        H_sub(row_here, 4) = sqrt_w * Jdt(1);
                        H_sub(row_here, 5) = sqrt_w * Jdt(2);

                        H_sub(row_here, 6) = sqrt_w * J_expo;

                        // 统计加权误差
                        patch_error += w * (res * res);
                        n_meas++;
                    }
                }

                // 存储该点的总误差（仅统计用）
                visual_submap->errors[i_pt] = patch_error;

#pragma omp atomic
                error += patch_error;
            } // end for i_pt

            // 处理完所有点后，移动 row_offset_cam
            row_offset_cam += (total_points * patch_size_total);
        } // end for cam_idx

        if (n_meas > 0) {
            error /= (float)n_meas;
        }
        compute_jacobian_time += (omp_get_wtime() - t1);

        // ------ 高斯牛顿 / EKF 更新 ------
        double t3 = omp_get_wtime();

        if (error <= last_error)
        {
            old_state = (*state);
            last_error = error;

            // 构造 H^T * H, 并叠加先验
            auto H_sub_T = H_sub.transpose();
            H_T_H.setZero();
            G.setZero();

            // 这里假设 7x7 状态: [3 rot, 3 trans, 1 expo]
            H_T_H.block<7,7>(0,0) = H_sub_T * H_sub;

            MD(DIM_STATE, DIM_STATE) K_1 =
                    (H_T_H + (state->cov / img_point_cov).inverse()).inverse();

            auto HTz = H_sub_T * z;

            // 预测 - 当前
            auto vec = (*state_propagat) - (*state);
            G.block<DIM_STATE, 7>(0,0) =
                    K_1.block<DIM_STATE, 7>(0,0) * H_T_H.block<7,7>(0,0);

            MD(DIM_STATE, 1) solution =
                    -K_1.block<DIM_STATE,7>(0,0) * HTz
                    + vec
                    - G.block<DIM_STATE,7>(0,0) * vec.block<7,1>(0,0);

            // 将解增量加回 state
            (*state) += solution;

            // 判断收敛：旋转增量 & 平移增量 是否足够小
            auto rot_add = solution.block<3,1>(0,0);
            auto t_add   = solution.block<3,1>(3,0);
            if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.f < 0.001f))
            {
                EKF_end = true;
            }
        }
        else
        {
            // 若本次误差反而变大，则回退
            (*state) = old_state;
            EKF_end = true;
        }

        update_ekf_time += (omp_get_wtime() - t3);

        // 若已收敛或到达最大迭代次数，退出
        if (iteration == max_iterations - 1 || EKF_end)
            break;
    }

    ROS_INFO("[updateState] Finished iterations with Huber, last error: %f", last_error);
}

void VIOManager::updateFrameState(StatesGroup state) {
    // 从状态中提取旋转和平移
    M3D Rwi(state.rot_end);
    V3D Pwi(state.pos_end);

    // 确保相机外参向量与相机数量一致
    assert(Rci_vec.size() == cams.size());
    assert(Pci_vec.size() == cams.size());

    // 确保 new_frame_->T_f_w_ 已经被初始化为与相机数量一致的向量
    if (new_frame_->T_f_w_.size() != cams.size()) {
        new_frame_->T_f_w_.resize(cams.size(), SE3());
    }

    // 遍历所有相机，计算并更新每台相机的位姿
    for (int cam_idx = 0; cam_idx < cams.size(); cam_idx++) {
        // 计算当前相机相对于 IMU (或世界) 的旋转和平移
        M3D Rcw = Rci_vec[cam_idx] * Rwi.transpose();

        V3D Pcw = -Rci_vec[cam_idx] * Rwi.transpose() * Pwi + Pci_vec[cam_idx];

        // 更新 new_frame_->T_f_w_ 中对应相机的位姿
        new_frame_->T_f_w_[cam_idx] = SE3(Rcw, Pcw);
    }
}
void VIOManager::plotTrackedPoints() {
    // 确保有点需要绘制
    int total_points = visual_submap->voxel_points.size();
    if (total_points == 0) return;

    // 获取相机数量
    int num_cams = cams.size();

    // 根据相机数量调整拼接布局：尽量接近正方形
    int num_rows = std::ceil(std::sqrt(num_cams));
    int num_cols = std::ceil(num_cams / (float)num_rows);

    // 假设所有相机图像尺寸一致，取第一幅图的尺寸
    int img_width = imgs_rgb[0].cols;
    int img_height = imgs_rgb[0].rows;

    // 使用全局变量 panorama_image，重新初始化它
    panorama_image.create(num_rows * img_height, num_cols * img_width, CV_8UC3);
    panorama_image.setTo(cv::Scalar(0, 0, 0));  // 填充黑色背景

    // 遍历每个相机，先在各自的图像上绘制点
    for (size_t camid = 0; camid < num_cams; ++camid) {
        // 克隆当前相机的彩色图像（BGR8）
        cv::Mat img = imgs_rgb[camid].clone();

        // 遍历所有点，在当前相机图像上绘制点
        for (int i = 0; i < total_points; i++) {
            VisualPoint* pt = visual_submap->voxel_points[i];
            if (pt == nullptr) continue; // 跳过空点

            // 将点从世界坐标系投影到当前相机的图像坐标系
            V2D pc = new_frame_->w2c(pt->pos_, camid);

            // 检查投影点是否在当前图像范围内
            if (pc[0] < 0 || pc[0] >= img.cols || pc[1] < 0 || pc[1] >= img.rows) {
                continue;
            }

            // 根据误差判断点为内点（绿色）或外点（蓝色）
            if (visual_submap->errors[i] <= visual_submap->propa_errors[i]) {
                cv::circle(img, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(0, 255, 0), -1, 8);
            } else {
                cv::circle(img, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(255, 0, 0), -1, 8);
            }

        }

        // 根据camid计算该图像在全景图中的位置
        int row = camid / num_cols;
        int col = camid % num_cols;
        cv::Rect roi(col * img_width, row * img_height, img_width, img_height);

        // 将处理好的图像复制到全景图的相应位置
        img.copyTo(panorama_image(roi));
    }

}


V3F VIOManager::getInterpolatedPixel(cv::Mat img, V2D pc) {
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0]);
    const int v_ref_i = floorf(pc[1]);
    const float subpix_u_ref = (u_ref - u_ref_i);
    const float subpix_v_ref = (v_ref - v_ref_i);
    const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    uint8_t *img_ptr = (uint8_t *) img.data + ((v_ref_i) * width + (u_ref_i)) * 3;
    float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] + w_ref_bl * img_ptr[width * 3] +
              w_ref_br * img_ptr[width * 3 + 0 + 3];
    float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] + w_ref_bl * img_ptr[1 + width * 3] +
              w_ref_br * img_ptr[width * 3 + 1 + 3];
    float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] + w_ref_bl * img_ptr[2 + width * 3] +
              w_ref_br * img_ptr[width * 3 + 2 + 3];
    V3F pixel(B, G, R);
    return pixel;
}

void VIOManager::dumpDataForColmap() {
    static int cnt = 1; // 静态计数器，用于命名图像文件
    std::ostringstream ss;
    ss << std::setw(5) << std::setfill('0') << cnt;
    std::string cnt_str = ss.str();
    std::string image_path = std::string(ROOT_DIR) + "Log/Colmap/images/" + cnt_str + ".png";

    // 确定 cam0 的索引
    const size_t cam0_idx = 0;

    // 检查 cam0 是否存在
    if (cam0_idx >= cams.size()) {
        std::cerr << "Error: cam0 index out of range." << std::endl;
        return;
    }

    // 获取 cam0 的图像
    cv::Mat img_rgb_cam0 = imgs_rgb[0];

    // 去畸变图像
    cv::Mat img_rgb_undistort_cam0;

    dynamic_cast<vk::PinholeCamera*>(cams[cam0_idx])->undistortImage(img_rgb_cam0, img_rgb_undistort_cam0);

    // 保存去畸变后的图像
    cv::imwrite(image_path, img_rgb_undistort_cam0);

    // 获取 cam0 的位姿
    SE3 cam0_pose = new_frame_->T_f_w_[cam0_idx];

    // 将旋转矩阵转换为四元数
    Eigen::Quaterniond q(cam0_pose.rotation_matrix());

    // 获取平移向量
    Eigen::Vector3d t = cam0_pose.translation();

    // 写入 COLMAP 的 images.txt 文件
    // 格式为:
    // IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID IMAGE_NAME
    fout_colmap << cnt << " "
                << std::fixed << std::setprecision(6)  // 保证浮点数精度为6位
                << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
                << t.x() << " " << t.y() << " " << t.z() << " "
                << 1 << " "  // CAMERA_ID (假设 cam0 对应 COLMAP 的 CAMERA_ID 为1)
                << cnt_str << ".png" << std::endl;

    // COLMAP 需要的额外行（可能是相机的方向，视具体格式而定）
    // 这里假设相机的方向是向下 Z 轴，即 [0.0 0.0 -1.0]
    fout_colmap << "0.0 0.0 -1.0" << std::endl;

    // 可选的调试输出
    std::cout << "Dumping data for cam0:" << std::endl;
    std::cout << "Image path: " << image_path << std::endl;
    std::cout << "Camera pose (quaternion): " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << std::endl;
    std::cout << "Camera pose (translation): " << t.transpose() << std::endl;

    cnt++; // 计数器递增
}


void VIOManager::processFrame(const std::vector<cv::Mat> &imgs,
                              std::vector<pointWithVar> &pg,
                              const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map,
                              double img_time)
{
    // 1) 校验 imgs.size() 与 cams.size() 是否匹配
    if (imgs.size() != cams.size()) {
        ROS_ERROR( "[ VIO ] Error: input imgs.size() != cams.size().  is  %zu %zu", imgs.size(), cams.size());
        return;
    }
    // 2) 对每个相机做一些图像预处理
    imgs_cp.resize(cams.size());
    imgs_rgb.resize(cams.size());
    for (size_t i = 0; i < cams.size(); i++) {
        if (imgs[i].empty()) {
            ROS_WARN_STREAM("Camera " << i << " image is empty, skipping.");
            continue;
        }

        // 检查分辨率是否需要 resize
        if (width != imgs[i].cols || height != imgs[i].rows) {
            cv::Mat resized;
            cv::resize(imgs[i], resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
            imgs_cp[i]  = resized.clone();
            imgs_rgb[i] = resized.clone();
        } else {
            imgs_cp[i]  = imgs[i].clone();
            imgs_rgb[i] = imgs[i].clone();
        }

        // 若是彩色，转灰度
        if (imgs_cp[i].channels() == 3) {
            cv::cvtColor(imgs_cp[i], imgs_cp[i], cv::COLOR_BGR2GRAY);
        }
    }
    resetGrid();
// 3) 构造/更新当前帧
    new_frame_.reset(new Frame(cams, imgs_cp)); // Frame 构造函数支持多相机
    updateFrameState(*state);

    double t1 = omp_get_wtime();

// 5) 从视觉稀疏地图检索可见点（多相机）
    ROS_INFO("Retrieving visible points from visual sparse map.");
    retrieveFromVisualSparseMap(imgs_cp, pg, feat_map);
    double t2 = omp_get_wtime();


    // 6) 计算雅可比并更新 EKF
    ROS_INFO("[ VIO ] Computing Jacobian and updating EKF");
    computeJacobianAndUpdateEKF(imgs_cp);
    double t3 = omp_get_wtime();

    // 7) 生成新的视觉地图点
    ROS_INFO("[ VIO ] Generating visual map points");
    generateVisualMapPoints(imgs_cp, pg);
    double t4 = omp_get_wtime();

    // 8) 可视化
    //ROS_INFO("[ VIO ] Visualizing tracked points");
    plotTrackedPoints();
    if (plot_flag)
        projectPatchFromRefToCur(feat_map);
    double t5 = omp_get_wtime();

    // 9) 更新已有地图点的属性
    ROS_INFO("[ VIO ] Updating visual map points");
    updateVisualMapPoints(imgs_cp);
    double t6 = omp_get_wtime();

    ROS_INFO("[ VIO ] Updating reference patch");
    updateReferencePatch(feat_map);
    double t7 = omp_get_wtime();

    if (colmap_output_en)
        dumpDataForColmap();

    frame_count++;
    ave_total = ave_total * (frame_count - 1) / frame_count + (t7 - t1 - (t5 - t4)) / frame_count;

    // ---- 打印时间统计 ----
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m|                         VIO Time (MultiCam)                 |\033[0m\n");
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size", feat_map.size());
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "retrieveFromVisualSparseMap", t2 - t1);
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "computeJacobianAndUpdateEKF", t3 - t2);
    printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian", compute_jacobian_time);
    printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF", update_ekf_time);
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "generateVisualMapPoints", t4 - t3);
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateVisualMapPoints", t6 - t5);
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateReferencePatch", t7 - t6);
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time", t7 - t1 - (t5 - t4));
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time", ave_total);
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
}


