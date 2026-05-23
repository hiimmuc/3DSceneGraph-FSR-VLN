#ifndef ODOM_LIB_H
#define ODOM_LIB_H
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <deque>
#include <mutex>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <utils/utils.h>
#include <nav_msgs/msg/path.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/voxel_grid_covariance.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>  // 补上 ICP 头
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
# define M_PI		3.14159265358979323846	/* pi */

class RelocRobotOdom
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  rclcpp::Node::SharedPtr node;
  RelocRobotOdom(std::string node_name, const rclcpp::NodeOptions &options) : node(std::make_shared<rclcpp::Node>(node_name, options))
  {
    sub_lidar_ = this->node->create_subscription<livox_ros_driver2::msg::CustomMsg>(
        "/livox/lidar", rclcpp::SensorDataQoS(),
        std::bind(&RelocRobotOdom::livoxCallback, this, std::placeholders::_1));

    sub_odom_ = this->node->create_subscription<nav_msgs::msg::Odometry>(
        "/robot_odom", 2000,
        std::bind(&RelocRobotOdom::odomCallback, this, std::placeholders::_1));

    pub_deskewed_ = this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/undistort_cloud", 10);
    pub_odometry_ = this->node->create_publisher<nav_msgs::msg::Odometry>(
        "/aft_mapped_to_init", 10);
    pub_path_ = this->node->create_publisher<nav_msgs::msg::Path>("/path", 10);
    path.header.stamp = this->node->now();
    path.header.frame_id = "camera_init";
    // p_wheel_to_imu
    this->node->get_parameter("wheel.p_wheel_to_imu", p_wheel_to_imu_);
    this->node->get_parameter("wheel.R_wheel_to_imu", R_wheel_to_imu_);
    std::cout<<"p_wheel_to_imu: "<<p_wheel_to_imu_[0]<<" "
             <<p_wheel_to_imu_[1]<<" "
             <<p_wheel_to_imu_[2]<<std::endl;
    T_wheel_to_imu_.block<3,3>(0,0) = Eigen::Map<const Eigen::Matrix3d>(R_wheel_to_imu_.data());
    T_wheel_to_imu_.block<3,1>(0,3) = Eigen::Map<const Eigen::Vector3d>(p_wheel_to_imu_.data());
    voxel_grid_.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
    prev_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    cloud_curr_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    cloud_curr_ds_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    RCLCPP_INFO(this->node->get_logger(), "✅ Livox deskew node started (using odometry)");
  }
  void run() {
    rclcpp::spin(this->node);
  }
public:
  struct OdomPose
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    rclcpp::Time stamp;
    Eigen::Vector3d pos;
    Eigen::Quaterniond rot;
  };

  std::deque<OdomPose, Eigen::aligned_allocator<OdomPose>> odom_buff_, odom_buff_tmp_;
  std::mutex mtx_odom_;
  std::vector<double> p_wheel_to_imu_;
  std::vector<double> R_wheel_to_imu_;
  std::deque<livox_ros_driver2::msg::CustomMsg::ConstSharedPtr> cloud_queue_;

  Eigen::Matrix4d T_wheel_to_imu_ = Eigen::Matrix4d::Identity();
  float leaf_size_ = 0.1f;

  nav_msgs::msg::Odometry lidar_odom_;
  geometry_msgs::msg::PoseStamped msg_body_pose;
  nav_msgs::msg::Path path;

  rclcpp::Time base_time_;
  Eigen::Matrix4d T_lidar_ = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_first_lidar_ = Eigen::Matrix4d::Identity();
  bool is_first_lidar_ = true;

  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_lidar_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_deskewed_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;

  pcl::PointCloud<pcl::PointXYZI>::Ptr prev_cloud_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_curr_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_curr_ds_;
  bool has_prev_cloud_{false};
  rclcpp::Time prev_base_time_;
  Eigen::Matrix4d T_prev_body_ = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_prev_lidar_ = Eigen::Matrix4d::Identity();

  pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_;

  // =============== Odom Buffer ===============
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    
    mtx_odom_.lock();
    OdomPose pose;
    pose.stamp = msg->header.stamp;
    pose.pos = Eigen::Vector3d(msg->pose.pose.position.x,
                               msg->pose.pose.position.y,
                               msg->pose.pose.position.z);
    pose.rot = Eigen::Quaterniond(msg->pose.pose.orientation.w,
                                  msg->pose.pose.orientation.x,
                                  msg->pose.pose.orientation.y,
                                  msg->pose.pose.orientation.z);
    if(is_first_lidar_){                              
      T_first_lidar_.block<3,3>(0,0) = pose.rot.toRotationMatrix();
      T_first_lidar_.block<3,1>(0,3) = pose.pos;
      is_first_lidar_ = false;
    }
    // convert to first lidar frame
    odom_buff_.emplace_back(pose);
    mtx_odom_.unlock();
  }

  // =============== Odom Interpolation ===============
  bool getInterpolatedPose(const rclcpp::Time &t, Eigen::Vector3d &pos, Eigen::Quaterniond &rot)
  {
    if (odom_buff_tmp_.size() < 2)
      return false;

    for (size_t i = 0; i < odom_buff_tmp_.size() - 1; ++i)
    {
      if (t >= odom_buff_tmp_[i].stamp && t <= odom_buff_tmp_[i + 1].stamp)
      {
        double ratio = (t - odom_buff_tmp_[i].stamp).seconds() /
                       (odom_buff_tmp_[i + 1].stamp - odom_buff_tmp_[i].stamp).seconds();
        pos = odom_buff_tmp_[i].pos * (1 - ratio) + odom_buff_tmp_[i + 1].pos * ratio;
        rot = odom_buff_tmp_[i].rot.slerp(ratio, odom_buff_tmp_[i + 1].rot);
        return true;
      }
    }
    return false;
  }
  void livoxCallback(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg)
  {
    cloud_queue_.emplace_back(msg);
    mtx_odom_.lock();
    rclcpp::Time cloud_head_time(cloud_queue_.front()->header.stamp);
    if(cloud_head_time + rclcpp::Duration::from_seconds(0.1) > odom_buff_.back().stamp){
      odom_buff_.clear();
      mtx_odom_.unlock();
      return;
    }
    mtx_odom_.unlock();
    processCloud(cloud_queue_.front());
    cloud_queue_.pop_front();
  }
  // =============== Livox 点云回调 ===============
  void processCloud(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg)
  {
    if (msg->point_num == 0)
      return;

    base_time_ = msg->header.stamp;
    odom_buff_tmp_.clear();
    mtx_odom_.lock();
    while(!odom_buff_.empty() && odom_buff_.front().stamp < base_time_-rclcpp::Duration::from_seconds(1.0)) odom_buff_.pop_front();
    odom_buff_tmp_ = odom_buff_;
    mtx_odom_.unlock();
    std::cout<<"odom buffer size: " << odom_buff_tmp_.size() << std::endl;

    RCLCPP_WARN(this->node->get_logger(), "Deskewing point cloud with base time %.3f", base_time_.seconds());
    RCLCPP_WARN(this->node->get_logger(), "Start Odometry %.3f", odom_buff_tmp_.front().stamp.seconds());
    RCLCPP_WARN(this->node->get_logger(), "End Odometry %.3f", odom_buff_tmp_.back().stamp.seconds());
    Eigen::Vector3d base_pos;
    Eigen::Quaterniond base_rot;
    if(!getInterpolatedPose(base_time_, base_pos, base_rot))
    {
      RCLCPP_WARN(this->node->get_logger(), "No odom for base time %.3f", base_time_.seconds());
      return;
    }
    // // 计算本帧的最大偏移时间（纳秒），并计算帧尾时间
    // uint64_t max_offset_ns = 0;
    // for (uint32_t i = 0; i < msg->point_num; ++i) {
    //   if (msg->points[i].offset_time > max_offset_ns) max_offset_ns = msg->points[i].offset_time;
    // }
    // // 若所有偏移都是 0，则按原方式处理（无运动）
    // rclcpp::Time end_time = base_time_;
    // if (max_offset_ns > 0) {
    //   end_time = base_time_ + rclcpp::Duration::from_seconds(static_cast<double>(max_offset_ns) / 1e9);
    // }

    // // 插值出帧尾位姿（只调用一次）
    // Eigen::Vector3d end_pos = base_pos;
    // Eigen::Quaterniond end_rot = base_rot;
    // if (max_offset_ns > 0) {
    //   if (!getInterpolatedPose(end_time, end_pos, end_rot)) {
    //     RCLCPP_WARN(this->node->get_logger(), "No odom for end time %.3f", end_time.seconds());
    //     return;
    //   }
    // }
    // 当前帧（IMU基准系）PCL 点云
     // 相对变换：从点时刻 -> 起始时刻
    Eigen::Matrix4d T_base = Eigen::Matrix4d::Identity();
    T_base.block<3, 3>(0, 0) = base_rot.toRotationMatrix();
    T_base.block<3, 1>(0, 3) = base_pos;
    cloud_curr_->points.resize(msg->point_num);
  // #pragma omp parallel for num_threads(2) // 将 4 替换为所需线程数
    for (uint32_t i = 0; i < msg->point_num; ++i)
    {
      const auto &p = msg->points[i];
      if (p.x*p.x + p.y*p.y + p.z*p.z < 0.01) continue;
    
      // 计算相对时间比例（0..1）
      // double ratio = 0.0;
      // if (max_offset_ns > 0) ratio = static_cast<double>(p.offset_time) / static_cast<double>(max_offset_ns);
      // std::cout << "Point " << i << " offset time: " << p.offset_time << " ratio: " << ratio << std::endl;
      // 线性插值平移，slerp 旋转
      // Eigen::Vector3d pos_i = base_pos * (1.0 - ratio) + end_pos * ratio;
      // Eigen::Quaterniond rot_i = base_rot.slerp(ratio, end_rot);
      Eigen::Vector3d pos_i;
      Eigen::Quaterniond rot_i;
      rclcpp::Time point_time = base_time_ + rclcpp::Duration::from_seconds(static_cast<double>(p.offset_time) / 1e9);
      if(!getInterpolatedPose(point_time, pos_i, rot_i)) continue;

      Eigen::Matrix4d T_i = Eigen::Matrix4d::Identity();
      T_i.block<3, 3>(0, 0) = rot_i.toRotationMatrix();
      T_i.block<3, 1>(0, 3) = pos_i;
      Eigen::Matrix4d T_rel = T_base.inverse() * T_i;
      T_rel = T_wheel_to_imu_ * T_rel * T_wheel_to_imu_.inverse();

      Eigen::Vector4d pt_raw(p.x, p.y, p.z, 1.0);
      Eigen::Vector4d pt_corrected = T_rel * pt_raw;
      pcl::PointXYZI q;
      q.x = static_cast<float>(pt_corrected.x());
      q.y = static_cast<float>(pt_corrected.y());
      q.z = static_cast<float>(pt_corrected.z());
      q.intensity = 0.f;
      cloud_curr_->points[i] = q;
    }
    cloud_curr_->width  = cloud_curr_->points.size();
    cloud_curr_->height = 1;
    cloud_curr_->is_dense = true;
    cloud_curr_ds_->clear();
    RCLCPP_WARN(this->node->get_logger(), "Current cloud points: %zu", cloud_curr_->size());
    voxel_grid_.setInputCloud(cloud_curr_);
    voxel_grid_.filter(*cloud_curr_ds_);
    RCLCPP_WARN(this->node->get_logger(), "Downsampled cloud points: %zu", cloud_curr_ds_->size());
    // =============== 帧间 ICP 修正里程计 ===============
    bool icp_ok = false;
    Eigen::Matrix4d delta_pred_world = T_prev_body_.inverse() * T_base;
    Eigen::Matrix4d init_guess_imu = T_wheel_to_imu_ * delta_pred_world * T_wheel_to_imu_.inverse();
    // has_prev_cloud_ = false;
    if (has_prev_cloud_ && prev_cloud_->size() > 200 && cloud_curr_ds_->size() > 200)
    {
      pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
      icp.setInputSource(cloud_curr_ds_);
      icp.setInputTarget(prev_cloud_);
      icp.setMaxCorrespondenceDistance(0.2);
      icp.setMaximumIterations(50);
      icp.setTransformationEpsilon(1e-6);
      icp.setEuclideanFitnessEpsilon(1e-6);
      pcl::PointCloud<pcl::PointXYZI> aligned;
      icp.align(aligned, init_guess_imu.cast<float>());
      if (icp.hasConverged() && icp.getFitnessScore() < 0.1){
        // // 只应用 ICP 得到的航向（yaw）修正，保留初始平移
        Eigen::Matrix4d T_icp_imu_full = icp.getFinalTransformation().cast<double>();
        // Eigen::Matrix4d T_delta_trans = T_icp_imu_full * init_guess_imu.inverse();
        // Eigen::Matrix3d T_delta_rot = T_delta_trans.block<3,3>(0,0);
        // double roll, pitch, yaw;
        // yaw = std::atan2(T_delta_rot(1,0), T_delta_rot(0,0));
        // while (yaw > M_PI) yaw -= 2.0*M_PI;
        // while (yaw < -M_PI) yaw += 2.0*M_PI;
        // std::cout << "ICP yaw correction: " << yaw / M_PI * 180.0 << std::endl;
        // if(abs(yaw) > M_PI / 60.0) yaw = 0.0; // 过大修正忽略
        // Eigen::Affine3f delta_rot_affine = pcl::getTransformation(0.0, 0.0, 0.0, 0.0, 0.0, yaw);
        // Eigen::Matrix3d T_yaw = delta_rot_affine.matrix().block<3,3>(0,0).cast<double>();
        // Eigen::Matrix4d T_correction = init_guess_imu;
        // T_correction.block<3,3>(0,0) = T_yaw * init_guess_imu.block<3,3>(0,0);
        T_lidar_ = T_prev_lidar_ * T_icp_imu_full;
        // std::cout<<"ICP correction "<< T_correction <<std::endl;
        RCLCPP_INFO(this->node->get_logger(), "ICP ok: score=%.4f inliers=%zu/%zu", icp.getFitnessScore(), aligned.size(), cloud_curr_ds_->size());
      } else {
        RCLCPP_WARN(this->node->get_logger(), "ICP not converged");
      }
      
    }else{
        T_lidar_ = T_prev_lidar_ * init_guess_imu; 
    }
    T_lidar_(2, 3)= 0.0; // 固定高度
    // 发布去畸变后的点云
    sensor_msgs::msg::PointCloud2 cloud_out_msg;
    pcl::toROSMsg(*cloud_curr_ds_, cloud_out_msg);
    cloud_out_msg.header.frame_id = "camera_init";
    cloud_out_msg.header.stamp = base_time_;
    pub_deskewed_->publish(cloud_out_msg);    
    publish_odometry(pub_odometry_);
    publish_path(pub_path_);
    // 更新上一帧缓存
    *prev_cloud_ = *cloud_curr_ds_;
    has_prev_cloud_ = true;
    prev_base_time_ = base_time_;
    T_prev_body_= T_base;
    T_prev_lidar_ = T_lidar_;
  }
  template <typename T>
  void set_pose(T &out) {
    out.position.x = T_lidar_(0, 3);
    out.position.y = T_lidar_(1, 3);
    out.position.z = T_lidar_(2, 3);
    Eigen::Matrix3d R = T_lidar_.block<3, 3>(0, 0);
    Eigen::Quaterniond q(R);
    out.orientation.w = q.w();
    out.orientation.x = q.x();
    out.orientation.y = q.y();
    out.orientation.z = q.z();
  }
  void publish_path(
      const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr &pubPath) {
    set_pose(msg_body_pose.pose);
    msg_body_pose.header.stamp = this->node->now();
    msg_body_pose.header.frame_id = "camera_init";
    path.poses.emplace_back(msg_body_pose);
    pubPath->publish(path);
  }

  void publish_odometry(
      const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr
          &pubOdomAftMapped) {
    lidar_odom_.header.frame_id = "camera_init";
    lidar_odom_.child_frame_id = "aft_mapped";
    lidar_odom_.header.stamp = base_time_;
    set_pose(lidar_odom_.pose.pose);

    static std::shared_ptr<tf2_ros::TransformBroadcaster> br;
    br = std::make_shared<tf2_ros::TransformBroadcaster>(this->node);

    tf2::Transform transform;
    tf2::Quaternion q;
    transform.setOrigin(
        tf2::Vector3(T_lidar_(0, 3), T_lidar_(1, 3), T_lidar_(2, 3)));
    Eigen::Matrix3d R = T_lidar_.block<3, 3>(0, 0);
    Eigen::Quaterniond geoQuat(R);
    q.setW(geoQuat.w());
    q.setX(geoQuat.x());
    q.setY(geoQuat.y());
    q.setZ(geoQuat.z());
    transform.setRotation(q);

    br->sendTransform(geometry_msgs::msg::TransformStamped(createTransformStamped(
        transform, lidar_odom_.header.stamp, "camera_init", "aft_mapped")));
    pubOdomAftMapped->publish(lidar_odom_);
  }

};

int main(int argc, char **argv)
{

  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);
  RelocRobotOdom reloc("reloc_robot_odom", options);
  reloc.run();
  rclcpp::shutdown();
  return 0;
}
