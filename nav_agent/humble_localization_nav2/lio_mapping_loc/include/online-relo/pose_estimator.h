#pragma once

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/voxel_grid_covariance.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <deque>
#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include <thread>

#include "../common_lib.h"
#include "../multi-session/Incremental_mapping.hpp"
#include "../tool_color_printf.h"
#include "../utils/tictoc.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#ifdef CUDA_EN
#include "ndt/ndt_cuda.hpp"
#include "gicp/fast_vgicp_cuda.hpp"
#endif

#define MAX_TIME_DIFF 0.05  // seconds, max time difference
class pose_estimator {

 public:
  pose_estimator(rclcpp::Node::SharedPtr& node);
  ~pose_estimator() {}

  // Parameters
  std::string priorDir;
  std::string cloudTopic;
  std::string poseTopic;
  std::string cloudTopic_repub;
  std::string poseTopic_repub;
  float searchDis;
  int searchNum;
  float trustDis;
  rclcpp::Node::SharedPtr node;
  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subCloud;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subPose;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      subExternalPose;

  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloud;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubPose;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubPriorMap;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubPriorPath;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubInitCloud;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubReloWorldCloud;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRelocBodyCloud;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubNearCloud;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      pubMeasurementEdge;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;

  // Point clouds
  pcl::PointCloud<PointTypeXYZI>::Ptr priorMap;
  pcl::PointCloud<PointTypeXYZI>::Ptr filterPriorMap;

  pcl::PointCloud<PointTypeXYZI>::Ptr priorPath;
  pcl::PointCloud<PointTypeXYZI>::Ptr reloCloudInMap;
  pcl::PointCloud<PointTypeXYZI>::Ptr cloudInBody;
  pcl::PointCloud<PointTypeXYZI>::Ptr initCloud;
  pcl::PointCloud<PointTypeXYZI>::Ptr initCloudInOdom;
  pcl::PointCloud<PointTypeXYZI>::Ptr nearCloud;
  pcl::PointCloud<PointTypeXYZI>::Ptr localMapDS;
  std::deque<pcl::PointCloud<PointTypeXYZI>::Ptr> initCloudBuffer;
  std::deque<Eigen::Matrix4f> initPoseBuffer;


  // Pose info
  PointTypePose externalPose;
  PointTypePose initPose;
  PointTypePose pose_zero;
  PointTypePose pose_ext;

  std::vector<double> extrinT_;
  std::vector<double> extrinR_;
  Eigen::Vector3d extrinT;
  Eigen::Matrix3d extrinR;
  int relo_interval;

  // KD-tree
  std::vector<int> idxVec;
  std::vector<float> disVec;
  pcl::KdTreeFLANN<PointTypeXYZI>::Ptr kdtreeGlobalMapPoses;

  std::vector<int> idxVec_copy;
  std::vector<float> disVec_copy;
  pcl::KdTreeFLANN<PointTypeXYZI>::Ptr kdtreeGlobalMapPoses_copy;

  pcl::VoxelGrid<PointTypeXYZI> downSizeFilterPub, downSizeFilterLocalMap,
      downSizeFilterCurrentCloud;
  pcl::VoxelGrid<PointTypeXYZI> downSizeInitCloud, downSizeFilterNearCloud;
  int idx = 1;
  std::deque<pcl::PointCloud<PointTypeXYZI>::Ptr> cloudBuffer;
  std::deque<double> cloudtimeBuffer;
  std::deque<Eigen::Affine3f> poseBuffer_6D;
  std::deque<double> posetimeBuffer;
  std::deque<PointTypeXYZI> poseBuffer_3D;
  std::ofstream fout_relo;
  std::mutex cloudBufferMutex;
  std::mutex poseBufferMutex;
  std::condition_variable sig_buffer;
  // PointTypePose currentPoseInOdom;
  Eigen::Affine3f currentPoseInOdom;
  Eigen::Affine3f lastPoseInOdom = Eigen::Affine3f::Identity();
  Eigen::Affine3f deltaPose;
  PointTypeXYZI currentPose3d;
  Eigen::Affine3f currentPoseInMap;
  Eigen::Affine3f lastPoseInMap = Eigen::Affine3f::Identity();

  pcl::PointCloud<PointTypeXYZI>::Ptr currentCloud,currentCloudDs;
  pcl::PointCloud<PointTypeXYZI>::Ptr currentCloudInMap;
  pcl::PointCloud<PointTypeXYZI>::Ptr currentCloudDsInMap;
  pcl::PointCloud<PointTypeXYZI>::Ptr currentCloudDsInOdom;
  double currentCloudTime;

  // Path and odometry
  double ld_time;
  nav_msgs::msg::Path path;
  nav_msgs::msg::Odometry odomAftMapped;
  geometry_msgs::msg::PoseStamped msg_body_pose;
  std::deque<PointTypePose> reloPoseBuffer;

  // Sessions and registration
  std::vector<MultiSession::Session> sessions;
  std::pair<int, float> detectResult;
  std::vector<int> invalid_idx;

#ifdef CUDA_EN
  fast_gicp::NDTCuda<PointTypeXYZI, PointTypeXYZI> ndtCuda;
  fast_gicp::FastVGICPCuda<PointTypeXYZI, PointTypeXYZI> fvgicpCuda;
#endif
  pcl::IterativeClosestPoint<PointTypeXYZI, PointTypeXYZI> icpPCL;
  pcl::NormalDistributionsTransform<PointTypeXYZI, PointTypeXYZI> ndtPCL;
  pcl::PointCloud<PointTypeXYZI>::Ptr ndt_alignedCloud;
  pcl::PointCloud<PointTypeXYZI>::Ptr currentLocalMapDS;

  // Flags
  bool buffer_flg = true;
  bool global_flg = false;
  bool external_flg = false;
  bool receive_ext_flg = false;
  bool sc_init_enable = false;
  bool sc_flg = false;
  int reg_mode_ = 0;
  int init_cloud_num_ = 5;

  float height;
  int cout_count = 0;
  int cout_count_ = 0;
  int sc_new = 1;
  int sc_old = -1;
  std::thread thread_loc_, thread_pub_;

  double localmap_res_,currentcloud_res_;

  // 多帧累积   
  bool accum_enable_ = false;
  int accum_window_size_ = 1;

  std::deque<pcl::PointCloud<PointTypeXYZI>::Ptr> ds_cloud_window_;
  std::deque<Eigen::Matrix4f> pose_odom_window_;  // 与 ds_cloud_window_ 对齐：T_odom_lidar

  pcl::PointCloud<PointTypeXYZI>::Ptr currentCloudDsAccum_;  // 累积输出（最新帧 LiDAR 系）
  pcl::PointCloud<PointTypeXYZI>::Ptr currentCloudReg_;      // 本次用于配准的输入（=Ds 或 Accum）
  // Methods
  void allocateMemory();

  void cloudCBK(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void poseCBK(const nav_msgs::msg::Odometry::SharedPtr msg);
  void externalCBK(
      const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);

  void run(rclcpp::Node::SharedPtr& node);
  void publish_cloud(rclcpp::Node::SharedPtr& node);
  void start_localization();
  bool easyToRelo(const PointTypeXYZI& pose3d);
  bool globalRelo();
  bool relocalization();
  // 方法A：使用旋转矩阵直接计算yaw（最稳定）
  float getYawFromTransform(const Eigen::Affine3f& tf);
  bool ndt_pcl(const pcl::PointCloud<PointTypeXYZI>::Ptr input_cloud,
                             const pcl::PointCloud<PointTypeXYZI>::Ptr target_cloud,
                             const Eigen::Matrix4f &init_pose,
                             Eigen::Matrix4f &output_pose);
  bool icp_pcl(const pcl::PointCloud<PointTypeXYZI>::Ptr input_cloud,
                             const pcl::PointCloud<PointTypeXYZI>::Ptr target_cloud,
                             const Eigen::Matrix4f &init_pose,
                             Eigen::Matrix4f &output_pose);
#ifdef CUDA_EN
  bool ndt_cuda(pcl::PointCloud<PointTypeXYZI>::Ptr input_cloud,
                            pcl::PointCloud<PointTypeXYZI>::Ptr target_cloud,
                            Eigen::Matrix4f &init_pose,
                            Eigen::Matrix4f &output_pose);
  bool vgicp_cuda(const pcl::PointCloud<PointTypeXYZI>::Ptr input_cloud,
                             const pcl::PointCloud<PointTypeXYZI>::Ptr target_cloud,
                             const Eigen::Matrix4f &init_pose,
                             Eigen::Matrix4f &output_pose);                            
#endif
  void lio_incremental();
  void publish_odometry(const Eigen::Affine3f& trans_aft);
  void publish_odometry(
      const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr& pub);
  void publish_path(
      const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr& pub);
  
  std::vector<double> initpose_prior_;  // Initial pose (x, y, yaw)
  bool enable_prior_pose_;  // Flag to enable prior pose initialization
  double ndt_score_threshold_;  // NDT fitness score threshold for valid registration
};
