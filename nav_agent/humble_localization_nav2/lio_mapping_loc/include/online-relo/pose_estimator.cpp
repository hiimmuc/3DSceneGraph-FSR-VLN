// ROS2 Humble version of pose_estimator.cpp (partial conversion -
// initialization only)
#include "pose_estimator.h"

pose_estimator::pose_estimator(rclcpp::Node::SharedPtr &node_) : node(node_) {
  allocateMemory();

  this->node->declare_parameter("relo.priorDir", std::string(" "));
  this->node->declare_parameter("relo.cloudTopic",
                                std::string("/cloud_registered"));
  this->node->declare_parameter("relo.poseTopic", std::string("/Odometry"));
  this->node->declare_parameter("relo.searchDis", 10.0);
  this->node->declare_parameter("relo.searchNum", 3);
  this->node->declare_parameter("relo.trustDis", 5.0);
  this->node->declare_parameter("relo.extrinsic_T", std::vector<double>());
  this->node->declare_parameter("relo.extrinsic_R", std::vector<double>());
  this->node->declare_parameter("relo.relo_interval", 10);
  // external_flg
  this->node->declare_parameter("relo.sc_init_enable", false);
  this->node->declare_parameter("relo.external_flg", false);
  // reg_mode_
  this->node->declare_parameter("relo.reg_mode", 0);
  // localmap_res_
  this->node->declare_parameter("relo.localmap_res", 0.2);
  // currentcloud_res_
  this->node->declare_parameter("relo.currentcloud_res", 0.2);
  // init_cloud_num_
  this->node->declare_parameter("relo.init_cloud_num", 5);

  // 简化版多帧累积参数
  this->node->declare_parameter("relo.accum_enable", false);
  this->node->declare_parameter("relo.accum_window_size", 5);
  this->node->declare_parameter<double>("relo.ndt_score_threshold", 0.3);

  this->node->get_parameter("relo.accum_enable", accum_enable_);
  this->node->get_parameter("relo.accum_window_size", accum_window_size_);
  if (accum_window_size_ < 1) accum_window_size_ = 1;

  this->node->get_parameter("relo.priorDir", priorDir);
  this->node->get_parameter("relo.cloudTopic", cloudTopic);
  this->node->get_parameter("relo.poseTopic", poseTopic);
  this->node->get_parameter("relo.searchDis", searchDis);
  this->node->get_parameter("relo.searchNum", searchNum);
  this->node->get_parameter("relo.trustDis", trustDis);
  this->node->get_parameter("relo.extrinsic_T", extrinT_);
  this->node->get_parameter("relo.extrinsic_R", extrinR_);
  // relo_interval
  this->node->get_parameter("relo.relo_interval", relo_interval);
  // external_flg
  this->node->get_parameter("relo.external_flg", external_flg);
  // sc_init_enable
  this->node->get_parameter("relo.sc_init_enable", sc_init_enable);
  // reg_mode_
  this->node->get_parameter("relo.reg_mode", reg_mode_);
  // localmap_res_
  this->node->get_parameter("relo.localmap_res", localmap_res_);
  // currentcloud_res_
  this->node->get_parameter("relo.currentcloud_res", currentcloud_res_);
  // init_cloud_num_
  this->node->get_parameter("relo.init_cloud_num", init_cloud_num_);

  extrinT << VEC_FROM_ARRAY(extrinT_);
  extrinR << MAT_FROM_ARRAY(extrinR_);
  // std::cout << "extrinT: " << extrinT << "\n" << "extrinR: " << extrinR
  //           << std::endl;

  // Eigen::Matrix<double, 3, 1> euler_ext = RotMtoEuler(extrinR);
  Eigen::Matrix<double, 3, 1> euler_ext =
      extrinR.eulerAngles(0, 1, 2);  // roll, pitch, yaw
  pose_ext.x = extrinT(0);
  pose_ext.y = extrinT(1);
  pose_ext.z = extrinT(2);
  pose_ext.roll = euler_ext(0, 0);
  pose_ext.pitch = euler_ext(1, 0);
  pose_ext.yaw = euler_ext(2, 0);

  pose_zero.x = 0.0;
  pose_zero.y = 0.0;
  pose_zero.z = 0.0;
  pose_zero.roll = 0.0;
  pose_zero.pitch = 0.0;
  pose_zero.yaw = 0.0;
  currentCloudTime = 0.0;
  subCloud = this->node->create_subscription<sensor_msgs::msg::PointCloud2>(
      cloudTopic, 100,
      std::bind(&pose_estimator::cloudCBK, this, std::placeholders::_1));

  subPose = this->node->create_subscription<nav_msgs::msg::Odometry>(
      poseTopic, 500,
      std::bind(&pose_estimator::poseCBK, this, std::placeholders::_1));

  pubCloud =
      this->node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud", 10);
  pubPose = this->node->create_publisher<nav_msgs::msg::Odometry>("/pose", 10);

  fout_relo.open(priorDir + "relo_pose.txt", std::ios::out);

  subExternalPose =
      this->node
          ->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
              "/initialpose", 10,
              std::bind(&pose_estimator::externalCBK, this,
                        std::placeholders::_1));

  pubPriorMap = this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/prior_map", 10);
  pubPriorPath = this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/prior_path", 10);
  pubReloWorldCloud =
      this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
          "/relo_world_cloud", 10);
  pubRelocBodyCloud =
      this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
          "/reloc_body_cloud", 10);

  pubInitCloud = this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/init_cloud", 10);
  pubNearCloud = this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/near_cloud", 10);
  pubMeasurementEdge =
      this->node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "measurement", 10);
  pubPath = this->node->create_publisher<nav_msgs::msg::Path>("/eloc_path",10);
  RCLCPP_INFO(this->node->get_logger(), "rostopic is ok");

  sessions.push_back(MultiSession::Session(1, "priorMap", priorDir, true));
  *priorMap += *sessions[0].globalMap;
  *priorPath += *sessions[0].cloudKeyPoses3D;
  pcl::VoxelGrid<PointTypeXYZI> filter;
  filter.setLeafSize(1.0f, 1.0f, 1.0f);
  filter.setInputCloud(priorMap);
  filter.filter(*filterPriorMap);
  publishCloud(pubPriorMap, filterPriorMap, this->node->now(), "world");


  downSizeFilterPub.setLeafSize(3.0, 3.0, 3.0);
  downSizeFilterLocalMap.setLeafSize(localmap_res_, localmap_res_, localmap_res_);
  downSizeFilterCurrentCloud.setLeafSize(currentcloud_res_, currentcloud_res_, currentcloud_res_);
#ifdef CUDA_EN
  // vgicp_cuda
  fvgicpCuda.setResolution(0.5);
  fvgicpCuda.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);
  fvgicpCuda.setMaximumIterations(50);
  fvgicpCuda.setMaxCorrespondenceDistance(0.1);
  // ndt_cuda
  ndtCuda.setResolution(1.0);
  ndtCuda.setDistanceMode(fast_gicp::NDTDistanceMode::D2D);
  ndtCuda.setTransformationEpsilon(1e-6);
  ndtCuda.setEuclideanFitnessEpsilon(1e-6);
  ndtCuda.setMaximumIterations(50);
  ndtCuda.setStepSize(0.1);
  ndtCuda.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
#endif

  ndtPCL.setResolution(1.0);             // 体素分辨率(米)，可在 0.5~2.0 调整
  ndtPCL.setStepSize(0.1);                // line search 步长
  ndtPCL.setTransformationEpsilon(1e-4);  // 收敛阈值
  ndtPCL.setMaximumIterations(50);        // 迭代次数·
  ndtPCL.setMinPointPerVoxel(3);         // 每个体素的最小点数


  icpPCL.setMaxCorrespondenceDistance(0.1);
  icpPCL.setMaximumIterations(50);
  icpPCL.setTransformationEpsilon(1e-4);
  icpPCL.setEuclideanFitnessEpsilon(1e-4);

  height = priorPath->points[0].z;

  kdtreeGlobalMapPoses->setInputCloud(priorPath);
  kdtreeGlobalMapPoses_copy->setInputCloud(priorPath);
  RCLCPP_INFO(this->node->get_logger(), "load prior knowledge");

  invalid_idx.emplace_back(-1);

  this->node->declare_parameter("relo.initpose_prior", std::vector<double>{0.0, 0.0, 0.0});
  this->node->declare_parameter("relo.enable_prior_pose", true);
  this->node->declare_parameter("relo.ndt_score", 0.15);

  this->node->get_parameter("relo.initpose_prior", initpose_prior_);
  this->node->get_parameter("relo.enable_prior_pose", enable_prior_pose_);
  this->node->get_parameter("relo.ndt_score_threshold", ndt_score_threshold_);

  if (enable_prior_pose_) {
    RCLCPP_INFO(this->node->get_logger(), "Using prior pose for initialization: x=%.2f, y=%.2f, yaw=%.2f", 
                initpose_prior_[0], initpose_prior_[1], initpose_prior_[2]);
    pose_zero.x = initpose_prior_[0];
    pose_zero.y = initpose_prior_[1];
    pose_zero.yaw = initpose_prior_[2];
    receive_ext_flg = true;
  } else {
    RCLCPP_INFO(this->node->get_logger(), "Waiting for manual initialization via RViz2...");
  }

}

void pose_estimator::allocateMemory() {
  priorMap.reset(new pcl::PointCloud<PointTypeXYZI>());
  filterPriorMap.reset(new pcl::PointCloud<PointTypeXYZI>());
  priorPath.reset(new pcl::PointCloud<PointTypeXYZI>());
  reloCloudInMap.reset(new pcl::PointCloud<PointTypeXYZI>());
  cloudInBody.reset(new pcl::PointCloud<PointTypeXYZI>());
  initCloud.reset(new pcl::PointCloud<PointTypeXYZI>());
  initCloudInOdom.reset(new pcl::PointCloud<PointTypeXYZI>());
  nearCloud.reset(new pcl::PointCloud<PointTypeXYZI>());
  localMapDS.reset(new pcl::PointCloud<PointTypeXYZI>());
  kdtreeGlobalMapPoses.reset(new pcl::KdTreeFLANN<PointTypeXYZI>());
  kdtreeGlobalMapPoses_copy.reset(new pcl::KdTreeFLANN<PointTypeXYZI>());
  currentCloud.reset(new pcl::PointCloud<PointTypeXYZI>());
  currentCloudDs.reset(new pcl::PointCloud<PointTypeXYZI>());
  currentCloudInMap.reset(new pcl::PointCloud<PointTypeXYZI>());
  currentCloudDsInOdom.reset(new pcl::PointCloud<PointTypeXYZI>());
  ndt_alignedCloud.reset(new pcl::PointCloud<PointTypeXYZI>());
  currentLocalMapDS.reset(new pcl::PointCloud<PointTypeXYZI>());
  currentCloudDsAccum_.reset(new pcl::PointCloud<PointTypeXYZI>());
  currentCloudReg_.reset(new pcl::PointCloud<PointTypeXYZI>());
}

void pose_estimator::cloudCBK(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  pcl::PointCloud<PointTypeXYZI>::Ptr msgCloud(
      new pcl::PointCloud<PointTypeXYZI>());
  pcl::fromROSMsg(*msg, *msgCloud);
  if (msgCloud->empty()) return;
  msgCloud->width = msgCloud->points.size();
  msgCloud->height = 1;
  cloudBufferMutex.lock();
  cloudBuffer.emplace_back(msgCloud);
  cloudtimeBuffer.emplace_back(msg->header.stamp.sec +
                               msg->header.stamp.nanosec * 1e-9);
  cloudBufferMutex.unlock();
  sig_buffer.notify_all();
}

void pose_estimator::poseCBK(const nav_msgs::msg::Odometry::SharedPtr msg) {
  Eigen::Affine3f pose;
  Eigen::Vector4d q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                    msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
  quaternionNormalize(q);
  Eigen::Matrix3d rot = quaternionToRotation(q);
  pose.linear() = rot.cast<float>();
  pose.translation() << msg->pose.pose.position.x, msg->pose.pose.position.y,
      msg->pose.pose.position.z;
  double time_stamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

  poseBufferMutex.lock();
  poseBuffer_6D.push_back(pose);
  posetimeBuffer.push_back(time_stamp);
  poseBufferMutex.unlock();
  sig_buffer.notify_all();
}

void pose_estimator::externalCBK(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
  RCLCPP_INFO(this->node->get_logger(),
              "please set your external pose now ...");
  externalPose.x = msg->pose.pose.position.x;
  externalPose.y = msg->pose.pose.position.y;
  externalPose.z = 0.0;
  tf2::Quaternion q;
  tf2::fromMsg(msg->pose.pose.orientation, q);
  double roll, pitch, yaw;
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
  externalPose.roll = 0.0;
  externalPose.pitch = 0.0;
  externalPose.yaw = yaw;
  RCLCPP_INFO(this->node->get_logger(),
              "Get initial pose: %.6f %.6f %.6f %.6f %.6f %.6f", externalPose.x,
              externalPose.y, externalPose.z, externalPose.roll,
              externalPose.pitch, externalPose.yaw);
  receive_ext_flg = true;
}

// 方法A：使用旋转矩阵直接计算yaw（最稳定）
float pose_estimator::getYawFromTransform(const Eigen::Affine3f& tf) {
    Eigen::Matrix3f R = tf.rotation();
    return std::atan2(R(1,0), R(0,0));  // 从2D旋转矩阵提取角度
}

void pose_estimator::run(rclcpp::Node::SharedPtr &node) {
  rclcpp::Rate rate(50);
  if (!rclcpp::ok()) {
    RCLCPP_ERROR(node->get_logger(), "Node is not ok, exiting run method.");
    return;
  }
  while (rclcpp::ok()) {
    rclcpp::spin_some(this->node);

    // Check if there are enough clouds and poses in the buffers
    cloudBufferMutex.lock();
    poseBufferMutex.lock();
    if (cloudBuffer.empty() || poseBuffer_6D.empty()) {
      cloudBufferMutex.unlock();
      poseBufferMutex.unlock();
      rate.sleep();
      continue;
    }

    // Synchronize poseBuffer_6D and cloudBuffer
    while (!poseBuffer_6D.empty() && !cloudBuffer.empty() &&
           std::abs(posetimeBuffer.front() - cloudtimeBuffer.front()) >
               MAX_TIME_DIFF) {
      if (posetimeBuffer.front() < cloudtimeBuffer.front()) {
        RCLCPP_WARN(
            this->node->get_logger(),
            "Pose timestamp: %.5f is earlier than cloud timestamp: %.5f",
            posetimeBuffer.front(), cloudtimeBuffer.front());
        poseBuffer_6D.pop_front();
        posetimeBuffer.pop_front();
      } else {
        cloudBuffer.pop_front();
        cloudtimeBuffer.pop_front();
      }
    }
    // print time stamp of cloud and pose
    if (!poseBuffer_6D.empty() && !cloudtimeBuffer.empty()) {
      currentCloud = cloudBuffer.front();
      cloudBuffer.pop_front();
      currentCloudTime = cloudtimeBuffer.front();
      cloudtimeBuffer.pop_front();
      cloudBufferMutex.unlock();

      currentPoseInOdom = poseBuffer_6D.front();
      poseBuffer_6D.pop_front();
      posetimeBuffer.pop_front();
      poseBufferMutex.unlock();

      sig_buffer.notify_all();

    } else {
      RCLCPP_WARN(this->node->get_logger(),
                  "Buffers are empty, cannot print timestamps.");
      cloudBufferMutex.unlock();
      poseBufferMutex.unlock();
      sig_buffer.notify_all();
      continue;
    }

    // Initialize the pose if not done yet
    if (!global_flg) {
      if (cout_count_ < 1) {
        std::cout << ANSI_COLOR_RED
                  << "wait for global pose initialization ... "
                  << ANSI_COLOR_RESET << std::endl;
      }
      global_flg = globalRelo();
      cout_count_ = 1;
      lastPoseInMap = currentPoseInMap;
      lastPoseInOdom = currentPoseInOdom;
      continue;
    }

    pcl::PointCloud<PointTypeXYZI>::Ptr relo_pt =
        std::make_shared<pcl::PointCloud<PointTypeXYZI>>();

    // Lk to Lk-1
    deltaPose = lastPoseInOdom.inverse() * currentPoseInOdom;
    // predict current pose in map frame
    currentPoseInMap = lastPoseInMap * deltaPose;
    PointTypeXYZI currentPose3dInMap;
    currentPose3dInMap.x = currentPoseInMap.translation().x();
    currentPose3dInMap.y = currentPoseInMap.translation().y();
    currentPose3dInMap.z = currentPoseInMap.translation().z();
    relo_pt->points.emplace_back(currentPose3dInMap);

    currentCloudDs->clear();
    downSizeFilterCurrentCloud.setInputCloud(currentCloud);
    downSizeFilterCurrentCloud.filter(*currentCloudDs);
    std::cout << "current ds cloud in lidar size: " << currentCloud->points.size()
              << std::endl;
    // 1) 维护窗口：只缓存，不做重计算
    if (accum_enable_ && accum_window_size_ > 1 && currentCloudDs && !currentCloudDs->empty()) {
      pcl::PointCloud<PointTypeXYZI>::Ptr ds_copy(new pcl::PointCloud<PointTypeXYZI>());
      *ds_copy = *currentCloudDs;
      ds_cloud_window_.push_back(ds_copy);
      pose_odom_window_.push_back(currentPoseInOdom.matrix());

      while ((int)ds_cloud_window_.size() > accum_window_size_) ds_cloud_window_.pop_front();
      while ((int)pose_odom_window_.size() > accum_window_size_) pose_odom_window_.pop_front();
    } else {
      // 关闭累积时，避免窗口无限增长
      ds_cloud_window_.clear();
      pose_odom_window_.clear();
    }

    bool relo_success = true;
    const bool need_relo =
        ((!relo_pt->points.empty() && easyToRelo(relo_pt->points[0]) &&
          idx % relo_interval == 0 && currentCloudDs->points.size() > 0) ||
         idx < 20);
    // 初始化稳定后降低重定位频率
    if (need_relo) {
      // 2) 仅在重定位前构建配准输入
      currentCloudReg_ = currentCloudDs;  // 默认用单帧
      if (accum_enable_ && accum_window_size_ > 1 && ds_cloud_window_.size() >= 2) {
        currentCloudDsAccum_->clear();

        const Eigen::Matrix4f T_odom_lidar_latest = pose_odom_window_.back();

        pcl::PointCloud<PointTypeXYZI>::Ptr accum_raw(new pcl::PointCloud<PointTypeXYZI>());
        accum_raw->reserve(200000);

        for (size_t i = 0; i < ds_cloud_window_.size(); ++i) {
          const Eigen::Matrix4f T_odom_lidar_i = pose_odom_window_[i];
          const Eigen::Matrix4f T_latest_i = T_odom_lidar_latest.inverse() * T_odom_lidar_i; // i -> latest
          pcl::PointCloud<PointTypeXYZI> tmp;
          pcl::transformPointCloud(*ds_cloud_window_[i], tmp, T_latest_i);
          *accum_raw += tmp;
        }

        // 再体素一次，防止点数暴涨（复用 downSizeFilterCurrentCloud 的 leaf size）
        downSizeFilterCurrentCloud.setInputCloud(accum_raw);
        downSizeFilterCurrentCloud.filter(*currentCloudDsAccum_);

        if (!currentCloudDsAccum_->empty()) {
          currentCloudReg_ = currentCloudDsAccum_;
          std::cout << "use accumulated cloud for registration, size="
                    << currentCloudReg_->points.size()
                    << " window=" << ds_cloud_window_.size() << std::endl;
        }
      }
      relo_success = relocalization();
      relo_pt->clear();
      if (!relo_success) {
        lio_incremental();
      }
    } else {
      lio_incremental();
    }
    idx++;
    lastPoseInMap = currentPoseInMap;
    lastPoseInOdom= currentPoseInOdom;
    // 添加map系机器人当前位置x,y,yaw打印
    float yaw_angle = getYawFromTransform(currentPoseInMap);
    std::cout << "Robot position in map frame - x: " << currentPoseInMap.translation().x()
              << ", y: " << currentPoseInMap.translation().y()
              << ", yaw: " << yaw_angle << " rad"
              << std::endl;
    

    rate.sleep();
  }
}

#ifdef CUDA_EN
bool pose_estimator::ndt_cuda(pcl::PointCloud<PointTypeXYZI>::Ptr input_cloud,
                              pcl::PointCloud<PointTypeXYZI>::Ptr target_cloud,
                              Eigen::Matrix4f &init_pose,
                              Eigen::Matrix4f &output_pose){                              
  // // 统计耗时
  ndtCuda.clearSource();
  ndtCuda.clearTarget();
  ndtCuda.setInputSource(input_cloud);
  ndtCuda.setInputTarget(target_cloud);
  pcl::PointCloud<PointTypeXYZI>::Ptr alignedCloud(
      new pcl::PointCloud<PointTypeXYZI>());
  ndtCuda.align(*alignedCloud,init_pose);
  output_pose = ndtCuda.getFinalTransformation();
  std::cout << "NDT-CUDA has converged: " << ndtCuda.hasConverged()
            << " with score: " << ndtCuda.getFitnessScore() << std::endl;
  if (ndtCuda.getFitnessScore() > 0.1) {
    std::cout << "NDT-CUDA did not converge or fitness score too high, skip this "
                 "relo"
              << std::endl;
    return false;
  }
  return true;
}
bool pose_estimator::vgicp_cuda(const pcl::PointCloud<PointTypeXYZI>::Ptr input_cloud,
                             const pcl::PointCloud<PointTypeXYZI>::Ptr target_cloud,
                             const Eigen::Matrix4f &init_pose,
                             Eigen::Matrix4f &output_pose){                              
  // // 统计耗时
  fvgicpCuda.clearSource();
  fvgicpCuda.clearTarget();
  fvgicpCuda.setInputSource(input_cloud);
  fvgicpCuda.setInputTarget(target_cloud);
  pcl::PointCloud<PointTypeXYZI>::Ptr alignedCloud(
      new pcl::PointCloud<PointTypeXYZI>());
  fvgicpCuda.align(*alignedCloud,init_pose);
  output_pose = fvgicpCuda.getFinalTransformation();
  std::cout << "VGICP-CUDA has converged: " << fvgicpCuda.hasConverged()
            << " with score: " << fvgicpCuda.getFitnessScore() << std::endl;
  if (fvgicpCuda.getFitnessScore() > 0.1) {
    std::cout << "VGICP-CUDA did not converge or fitness score too high, skip this "
                 "relo"
              << std::endl;
    return false;
  }
  return true;
}
#endif
bool pose_estimator::icp_pcl(const pcl::PointCloud<PointTypeXYZI>::Ptr input_cloud,
                             const pcl::PointCloud<PointTypeXYZI>::Ptr target_cloud,
                             const Eigen::Matrix4f &init_pose,
                             Eigen::Matrix4f &output_pose){

  icpPCL.setInputSource(input_cloud);
  icpPCL.setInputTarget(target_cloud);
  pcl::PointCloud<PointTypeXYZI>::Ptr alignedCloud(
      new pcl::PointCloud<PointTypeXYZI>());
  icpPCL.align(*alignedCloud,init_pose);
  output_pose = icpPCL.getFinalTransformation();
  Eigen::Matrix3d rot = output_pose.block<3, 3>(0, 0).cast<double>();
  Eigen::Vector3d linear = output_pose.block<3, 1>(0, 3).cast<double>();
  std::cout << "PCL ICP has converged: " << icpPCL.hasConverged()
            << " with score: " << icpPCL.getFitnessScore() << std::endl;
  std::cout << "PCL ICP lidar to map transformation: " << linear.transpose() << " "
            << rot.eulerAngles(0, 1, 2).transpose() << std::endl;
  if (!icpPCL.hasConverged() || icpPCL.getFitnessScore() > 0.1) {
    std::cout << "PCL ICP did not converge or fitness score too high, skip this "
                 "relo"
              << std::endl;
    return false;
  }
  return true;
}
bool pose_estimator::ndt_pcl(const pcl::PointCloud<PointTypeXYZI>::Ptr input_cloud,
                             const pcl::PointCloud<PointTypeXYZI>::Ptr target_cloud,
                             const Eigen::Matrix4f &init_pose,
                             Eigen::Matrix4f &output_pose){
  if (!target_cloud || target_cloud->empty()) {
      std::cerr << "Error: target cloud is empty!" << std::endl;
      return false;
  }
  if (!input_cloud || input_cloud->empty()) {
      std::cerr << "Error: input cloud is empty!" << std::endl;
      return false;
  }
  ndtPCL.setInputSource(input_cloud);
  ndtPCL.setInputTarget(target_cloud);
  pcl::PointCloud<PointTypeXYZI>::Ptr alignedCloud(new pcl::PointCloud<PointTypeXYZI>());
  ndtPCL.align(*alignedCloud,init_pose);
  output_pose = ndtPCL.getFinalTransformation();
  Eigen::Matrix3d rot = output_pose.block<3, 3>(0, 0).cast<double>();
  Eigen::Vector3d linear = output_pose.block<3, 1>(0, 3).cast<double>();
  std::cout << "PCL NDT has converged: " << ndtPCL.hasConverged()
            << " with score: " << ndtPCL.getFitnessScore() << std::endl;
  std::cout << "PCL NDT lidar to map transformation: " << linear.transpose() << " "
            << rot.eulerAngles(0, 1, 2).transpose() << std::endl;
  if (!ndtPCL.hasConverged() || ndtPCL.getFitnessScore() > 0.15) {
    std::cout << "PCL NDT did not converge or fitness score too high, skip this "
              << std::endl;
    return false;
  }
  return true;
}

// 直接计算lidar to map的位姿
bool pose_estimator::relocalization() {
  std::cout << ANSI_COLOR_GREEN << "relo mode for frame: " << idx
            << ANSI_COLOR_RESET << std::endl;
  // *** 修复：创建全新的局部点云 ***
  // 不要使用成员变量 nearCloud 和 localMapDS 来存储当前帧的 target map
  pcl::PointCloud<PointTypeXYZI>::Ptr tempNearCloud(new pcl::PointCloud<PointTypeXYZI>());
  for (auto &it : idxVec) {
    *tempNearCloud +=
        *transformPointCloud(sessions[0].cloudKeyFrames[it].all_cloud,
                             &sessions[0].cloudKeyPoses6D->points[it]);
  }
  // 本次用于配准的输入云（单帧 or 多帧累积）
  pcl::PointCloud<PointTypeXYZI>::Ptr input_cloud = currentCloudReg_;
  if (!input_cloud) input_cloud = currentCloudDs;

  // downsize
  currentLocalMapDS->clear();
  downSizeFilterLocalMap.setInputCloud(tempNearCloud);
  downSizeFilterLocalMap.filter(*currentLocalMapDS);
  std::cout << "downsized local map size: " << currentLocalMapDS->points.size()
            << std::endl;
  if (currentLocalMapDS->points.size() < 1000) {
    std::cout << "local map size is too small, skip this relo" << std::endl;
    return false;
  }

  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  auto start = std::chrono::high_resolution_clock::now();
  bool ndt_success = false;
  // initialize NDT inputs from current downsampled cloud and local map
  if(reg_mode_==0){
    ndt_success = ndt_pcl(input_cloud, currentLocalMapDS, currentPoseInMap.matrix(), transform);
  }else if(reg_mode_==1){
    ndt_success = icp_pcl(input_cloud, currentLocalMapDS, currentPoseInMap.matrix(), transform);
#ifdef CUDA_EN
  }else if(reg_mode_==2){
    ndt_success = ndt_cuda(input_cloud, currentLocalMapDS, currentPoseInMap.matrix(), transform);
  }else if(reg_mode_==3){
    ndt_success = vgicp_cuda(input_cloud, currentLocalMapDS, currentPoseInMap.matrix(), transform);
#endif
  }else{
    RCLCPP_ERROR(this->node->get_logger(), "reg_mode_ is invalid or selected CUDA mode not available!");
    return false;
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::ofstream fout_time,fout_ndt_score;
  // fout_time.open(std::string(ROOT_DIR) + "Log/result/reloc_time.txt",
  //                std::ios::app);
  fout_ndt_score.open(std::string(ROOT_DIR) + "Log/result/ndt_score.txt",
                 std::ios::app);
  // if (!fout_time.is_open())
  //   RCLCPP_ERROR(this->node->get_logger(), "open fail\n");
  // fout_time << std::fixed << std::setprecision(6) << elapsed.count() * 1e3 << std::endl;
  
  fout_ndt_score << std::fixed << std::setprecision(6);
  double score = 1.0;
  if(reg_mode_ == 0) score = ndtPCL.getFitnessScore();
  else if(reg_mode_ == 1) score = icpPCL.getFitnessScore();
#ifdef CUDA_EN
  else if(reg_mode_ == 2) score = ndtCuda.getFitnessScore();
  else if(reg_mode_ == 3) score = fvgicpCuda.getFitnessScore();
#endif
  fout_ndt_score << score << std::endl;

  std::cout << "NDT registration time: " << elapsed.count() << " seconds"
            << std::endl;
  if (!ndt_success) {
    return false;
  }
  // lidar to map
  currentPoseInMap = Eigen::Affine3f(transform);
  // cloud in map frame
  reloCloudInMap->clear();
  pcl::transformPointCloud(*input_cloud, *reloCloudInMap, currentPoseInMap.matrix());
  publishCloud(pubReloWorldCloud, reloCloudInMap, this->node->now(), "world");

  // body pose in map
  Eigen::Affine3f lidar2body;
  lidar2body.translation() << 0.0, 0.0, 0.0;
  Eigen::Matrix3f extrinR;
  extrinR << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0;
  lidar2body.linear() = extrinR.cast<float>();

  // cloud in body frame for local planning 
  cloudInBody->clear();
  // transformPointCloud(currentCloud, lidar2body, cloudInBody);
  pcl::transformPointCloud(*currentCloud, *cloudInBody, lidar2body.matrix());
  publishCloud(pubRelocBodyCloud, cloudInBody,
               /*rclcpp::Time(currentCloudTime * 1e9)*/ this->node->now(),
               "base_link");

  // std::cout << "lidar to map transformation: " << currentPoseInMap.translation().x()
  //           << " " << currentPoseInMap.translation().y() << " "
  //           << currentPoseInMap.translation().z() << " "
  //           << currentPoseInMap.rotation().eulerAngles(0, 1, 2).transpose()
  //           << std::endl;
  publish_odometry(currentPoseInMap);
  publish_path(pubPath);

  return true;
}

void pose_estimator::lio_incremental() {
  std::cout << ANSI_COLOR_RED << "livo mode for frame: " << idx
            << ANSI_COLOR_RESET << std::endl;

  Eigen::Affine3f lidar2body;
  lidar2body.translation() << 0.0, 0.0, 0.0;
  Eigen::Matrix3f extrinR;
  extrinR << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0;
  lidar2body.linear() = extrinR.cast<float>();

  // cloud in body frame
  cloudInBody->clear();
  // transformPointCloud(currentCloud, lidar2body, cloudInBody);
  pcl::transformPointCloud(*currentCloud, *cloudInBody, lidar2body.matrix());
  publishCloud(pubRelocBodyCloud, cloudInBody,
               /*rclcpp::Time(currentCloudTime * 1e9)*/ this->node->now(),
               "base_link");
  
  // cloud in map frame
  reloCloudInMap->clear();
  // transformPointCloud(currentCloudDs, currentPoseInMap, reloCloudInMap);
  pcl::transformPointCloud(*currentCloudDs, *reloCloudInMap, currentPoseInMap.matrix());
  publishCloud(pubReloWorldCloud, reloCloudInMap, this->node->now(), "world");

  // std::cout << "livo transformation in map: " << currentPoseInMap.translation().x()
  //           << " " << currentPoseInMap.translation().y() << " "
  //           << currentPoseInMap.translation().z() << " "
  //           << currentPoseInMap.rotation().eulerAngles(0, 1, 2).transpose()
  //           << std::endl;
  publish_odometry(currentPoseInMap);
  publish_path(pubPath);
}
void pose_estimator::publish_odometry(const Eigen::Affine3f &trans_aft) {
  Eigen::Matrix<double, 3, 3> ang_rot = trans_aft.rotation().cast<double>();
  Eigen::Quaterniond quaternion(ang_rot);
  odomAftMapped.pose.pose.position.x = trans_aft.translation().x();
  odomAftMapped.pose.pose.position.y = trans_aft.translation().y();
  odomAftMapped.pose.pose.position.z = trans_aft.translation().z();
  odomAftMapped.pose.pose.orientation.x = quaternion.x();
  odomAftMapped.pose.pose.orientation.y = quaternion.y();
  odomAftMapped.pose.pose.orientation.z = quaternion.z();
  odomAftMapped.pose.pose.orientation.w = quaternion.w();
  publish_odometry(pubPose);
}

bool pose_estimator::easyToRelo(const PointTypeXYZI &pose3d) {
  idxVec.clear();
  disVec.clear();
  idxVec_copy.clear();
  disVec_copy.clear();

  // Perform radius search on the primary kdtree
  kdtreeGlobalMapPoses->radiusSearch(pose3d, searchDis, idxVec, disVec);

  // Check if the search results are valid
  if (!disVec.empty() && disVec[0] <= searchDis && idxVec.size() > searchNum) {
    return true;
  }
  // Perform radius search on the secondary kdtree with an extended radius
  kdtreeGlobalMapPoses_copy->radiusSearchT(pose3d, searchDis * 2.0, idxVec_copy,
                                           disVec_copy);

  if (idxVec_copy.size() > 4 && disVec[0] <= searchDis * 2.0) {
    std::cout << ANSI_COLOR_RED << "relo by secondary search with "
              << idxVec_copy.size() << " points" << ANSI_COLOR_RESET
              << std::endl;
    // If the secondary search yields enough results, return true
    idxVec = idxVec_copy;
    disVec = disVec_copy;
    return true;
  }
  return false;
}

bool pose_estimator::globalRelo() {
  int detectID = -1;
  static int cloud_count = 0;

  if (!sc_flg && sc_init_enable) {
    // 初始化时对点云进行累加，至少5帧 currentCloud在lidar系下
    pcl::PointCloud<PointTypeXYZI>::Ptr tmpCloud(new pcl::PointCloud<PointTypeXYZI>());
    pcl::copyPointCloud(*currentCloud, *tmpCloud);
    initCloudBuffer.emplace_back(tmpCloud);
    initPoseBuffer.emplace_back(currentPoseInOdom.matrix());
    if(initCloudBuffer.size() < init_cloud_num_) {
      std::cout << ANSI_COLOR_RED
                << "wait for more cloud frame for sc, current frame count: "
                << initCloudBuffer.size() << ANSI_COLOR_RESET << std::endl;
      return false;
    }
    // Transform each buffered cloud into the coordinate frame of the latest buffered pose and accumulate.
    initCloud->clear();
    if (!initPoseBuffer.empty()) {
      Eigen::Matrix4f latest_pose = initPoseBuffer.back();
      Eigen::Matrix4f latest_pose_inv;
      latest_pose_inv.block<3, 3>(0, 0) = latest_pose.block<3, 3>(0, 0).transpose();
      latest_pose_inv.block<3, 1>(0, 3) = -latest_pose_inv.block<3, 3>(0, 0) * latest_pose.block<3, 1>(0, 3);
      latest_pose_inv.row(3) << 0.0f, 0.0f, 0.0f, 1.0f;
      for (size_t i = 0; i < initCloudBuffer.size(); ++i) {
        Eigen::Matrix4f pose_i = initPoseBuffer[i];
        Eigen::Matrix4f T = latest_pose_inv * pose_i;
        pcl::PointCloud<PointTypeXYZI>::Ptr tmp(new pcl::PointCloud<PointTypeXYZI>());
        pcl::transformPointCloud(*initCloudBuffer[i], *tmp, T);
        *initCloud += *tmp;
      }
    } else {
      // Fallback: just accumulate raw clouds if no pose info (should not happen)
      for (auto &c : initCloudBuffer) {
        *initCloud += *c;
      }
    }
    // 清空缓存，后续使用 initCloud 作为初始化输入
    initCloudBuffer.clear();
    initPoseBuffer.clear();
    std::cout << ANSI_COLOR_GREEN << "global relo by sc ... "
              << ANSI_COLOR_RESET << std::endl;

    // Must be body frame when calculating scancontext
    Eigen::MatrixXd initSC = sessions[0].scManager.makeScancontext(*initCloud);
    Eigen::MatrixXd ringkey =
        sessions[0].scManager.makeRingkeyFromScancontext(initSC);
    Eigen::MatrixXd sectorkey =
        sessions[0].scManager.makeSectorkeyFromScancontext(initSC);
    std::vector<float> polarcontext_invkey_vec =
        ScanContext::eig2stdvec(ringkey);
    detectResult = sessions[0].scManager.detectClosestKeyframeID(
        0, invalid_idx, polarcontext_invkey_vec, initSC);
    detectID = detectResult.first;

    std::cout << " current cloud size: " << initCloud->points.size()
              << std::endl;
    std::cout << ANSI_COLOR_RED << "init relocalization by current SC id: " << 0
              << " in prior map's SC id: " << detectResult.first
              << " yaw offset: " << -detectResult.second << ANSI_COLOR_RESET
              << std::endl;
  }

  if (sc_init_enable && detectID > -1) {
    std::cout << ANSI_COLOR_GREEN << "init relo by scan context ... "
              << ANSI_COLOR_RESET << std::endl;
    std::cout << ANSI_COLOR_GREEN << "use prior frame " << detectID
              << " to relo init cloud ..." << ANSI_COLOR_RESET << std::endl;

    PointTypePose poseOffset =
        sessions[0].cloudKeyPoses6D->points[detectResult.first];
    std::cout << "sc prior frame pose in map: " << poseOffset.x << " " << poseOffset.y
              << " " << poseOffset.z << " " << poseOffset.roll << " "
              << poseOffset.pitch << " " << poseOffset.yaw << std::endl;
    
    // search nearby keyframes
    PointTypeXYZI tmp;
    tmp.x = poseOffset.x;
    tmp.y = poseOffset.y;
    tmp.z = poseOffset.z;
    idxVec.clear();
    disVec.clear();
    kdtreeGlobalMapPoses->nearestKSearch(tmp, searchNum * 2 , idxVec, disVec);

    // convert to map
    pcl::PointCloud<PointTypeXYZI>::Ptr tempNearCloud(new pcl::PointCloud<PointTypeXYZI>());
    for (int i = 0; i < idxVec.size(); i++) {
      *tempNearCloud +=
          *transformPointCloud(sessions[0].cloudKeyFrames[idxVec[i]].all_cloud,
                               &sessions[0].cloudKeyPoses6D->points[idxVec[i]]);
    }
    
    std::cout << ANSI_COLOR_GREEN << "get precise pose by NDT ... "
              << ANSI_COLOR_RESET << std::endl;
    std::cout << ANSI_COLOR_GREEN << "local map cloud size: " << tempNearCloud->points.size()
              << ANSI_COLOR_RESET << std::endl;
    std::cout << ANSI_COLOR_GREEN << "current init cloud size: " << initCloud->points.size()
              << ANSI_COLOR_RESET << std::endl;
    // current frame to loop frame by sc
    Eigen::Affine3f transCurFrame2PriorFrame =
        pcl::getTransformation(0.0, 0.0, 0.0, 0.0, 0.0, -detectResult.second);
    // loop frame to map 
    Eigen::Affine3f transPriorFrame2Map =
        pcl::getTransformation(poseOffset.x, poseOffset.y, poseOffset.z,
                               poseOffset.roll, poseOffset.pitch,
                               poseOffset.yaw);
    Eigen::Affine3f initPoseInMap =
        transPriorFrame2Map * transCurFrame2PriorFrame;         
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    bool ndt_success=ndt_pcl(initCloud, tempNearCloud, initPoseInMap.matrix(), transform); 

    reloCloudInMap->clear();
    pcl::transformPointCloud(*initCloud, *reloCloudInMap, transform);
    publishCloud(pubReloWorldCloud, reloCloudInMap, this->node->now(), "world");
    
    if (!ndt_success) {
      if(ndtPCL.getFitnessScore()>ndt_score_threshold_) invalid_idx.emplace_back(detectID);
      sc_flg = false;
      return false;
    } else {
      sc_flg = true;
    }
    // update current pose in map
    currentPoseInMap.matrix() = transform;
    global_flg = true;
    std::cout << ANSI_COLOR_GREEN << "init lidar to map pose: " << currentPoseInMap.translation().x() << " "
              << currentPoseInMap.translation().y() << " " << currentPoseInMap.translation().z() << std::endl;
    std::cout << ANSI_COLOR_GREEN
              << "init relocalization has been finished ... "
              << ANSI_COLOR_RESET << std::endl;
  } else if (external_flg && receive_ext_flg) {
    std::cout << ANSI_COLOR_GREEN << "init relo by external-pose ... "
              << ANSI_COLOR_RESET << std::endl;
    // 初始化时对点云进行累加，至少5帧 currentCloud在lidar系下
    cloud_count++;
    pcl::PointCloud<PointTypeXYZI>::Ptr tmpCloud(new pcl::PointCloud<PointTypeXYZI>());
    transformPointCloud(currentCloud, currentPoseInOdom, tmpCloud);
    *initCloudInOdom +=*tmpCloud;
    if(cloud_count < init_cloud_num_) {
      std::cout << ANSI_COLOR_RED
                << "wait for more cloud frame, current frame count: "
                << cloud_count << ANSI_COLOR_RESET << std::endl;
      return false;
    }
    cloud_count = 0;
    if (initCloudInOdom->points.size() < 2000) {
      std::cout << ANSI_COLOR_RED
                << "current cloud size is too small, wait for next frame ..."
                << ANSI_COLOR_RESET << std::endl;
      return false;
    }

    // odom to map
    // odom frame has been aligned to gravity direction
    if (enable_prior_pose_) {
      externalPose.x = initpose_prior_[0];
      externalPose.y = initpose_prior_[1];
      externalPose.yaw = initpose_prior_[2];
    }
    PointTypePose pose_offset;
    pose_offset.x = externalPose.x;
    pose_offset.y = externalPose.y;
    pose_offset.z = 0.0;
    pose_offset.roll = 0.0;
    pose_offset.pitch = 0.0;
    pose_offset.yaw = externalPose.yaw;
    Eigen::Affine3f init_odom2map = pcl::getTransformation(
        pose_offset.x, pose_offset.y, pose_offset.z, pose_offset.roll,
        pose_offset.pitch, pose_offset.yaw);

    // search nearby keyframes
    idxVec.clear();
    disVec.clear();
    PointTypeXYZI tmp;
    tmp.x = externalPose.x;
    tmp.y = externalPose.y;
    tmp.z = externalPose.z;
    kdtreeGlobalMapPoses->radiusSearchT(tmp, searchDis * 2, idxVec, disVec);
    
    // get local map for initialization
    pcl::PointCloud<PointTypeXYZI>::Ptr tempNearCloud(new pcl::PointCloud<PointTypeXYZI>());
    for (int i = 0; i < idxVec.size(); i++) {
      *tempNearCloud +=
          *transformPointCloud(sessions[0].cloudKeyFrames[idxVec[i]].all_cloud,
                               &sessions[0].cloudKeyPoses6D->points[idxVec[i]]);
    }

    std::cout << "local cloud size for initialization: " << tempNearCloud->points.size() << std::endl;
    std::cout << ANSI_COLOR_GREEN << "get precise pose by NDT ... "
              << ANSI_COLOR_RESET << std::endl;
    currentCloudDsInOdom->clear();
    downSizeInitCloud.setLeafSize(0.20, 0.20, 0.20);
    downSizeInitCloud.setInputCloud(initCloudInOdom);
    downSizeInitCloud.filter(*currentCloudDsInOdom);
    std::cout << "init cloud size: " << currentCloudDsInOdom->points.size() << std::endl;

    initCloudInOdom->clear();

    currentLocalMapDS->clear();
    downSizeFilterNearCloud.setLeafSize(0.20, 0.20, 0.20);
    downSizeFilterNearCloud.setInputCloud(tempNearCloud);
    downSizeFilterNearCloud.filter(*currentLocalMapDS);
    std::cout << "init local map size: " << currentLocalMapDS->points.size()
              << std::endl;
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    bool ndt_success=ndt_pcl(currentCloudDsInOdom, currentLocalMapDS, init_odom2map.matrix(), transform);

    // transform is from odom to map
    Eigen::Affine3f transformAffine;
    transformAffine.matrix() = transform;
    // convert current cloud in odom to map frame for visualization
    reloCloudInMap->clear();
    transformPointCloud(currentCloudDsInOdom, transformAffine, reloCloudInMap);
    publishCloud(pubReloWorldCloud, reloCloudInMap, this->node->now(), "world");
    Eigen::Matrix4f trans_diff =
        init_odom2map.matrix().inverse() * transform;
    float pos_diff = trans_diff.block<3, 1>(0, 3).norm();
    std::cout << "NDT delta pose difference: " << pos_diff << std::endl;
    float ndt_score = ndtPCL.getFitnessScore();
    std::cout << "ndt_score_threshold_ = " << ndt_score_threshold_ << std::endl;
    if (!ndt_success && ndt_score > ndt_score_threshold_) {
      std::cout << ANSI_COLOR_RED
                << "NDT registration failed, fitness score too high: "
                << ndtPCL.getFitnessScore() << ANSI_COLOR_RESET << std::endl;
      receive_ext_flg = false;
      std::cout << ANSI_COLOR_RED
                << "please update external pose to continue ..."
                << ANSI_COLOR_RESET << std::endl;
      return false;
    }

    std::cout << "NDT has converged: " << ndtPCL.hasConverged()
              << " with score: " << ndtPCL.getFitnessScore() << std::endl;
    sc_flg = true;

    // optimazed odom to map transformation
    Eigen::Matrix3d rot = transform.matrix().block<3, 3>(0, 0).cast<double>();
    Eigen::Vector3d linear =
        transform.matrix().block<3, 1>(0, 3).cast<double>();
    currentPoseInMap= transformAffine * currentPoseInOdom;
    global_flg = true;

    // std::cout << ANSI_COLOR_GREEN << "init lidar to map pose: " << currentPoseInMap.translation().x() << " "
    //           << currentPoseInMap.translation().y() << " " << currentPoseInMap.translation().z() << std::endl;
    float yaw_angle = getYawFromTransform(currentPoseInMap);
    std::cout << ANSI_COLOR_GREEN << "Robot position in map frame - x: " << currentPoseInMap.translation().x()
              << ", y: " << currentPoseInMap.translation().y()
              << ", yaw: " << yaw_angle << " rad"
              << std::endl;
              
    std::cout << ANSI_COLOR_GREEN
              << "init relocalization by external-pose has been finished ... "
              << ANSI_COLOR_RESET << std::endl;
  } else {
    sc_flg = false;
    return false;
  }
  return true;
}

void pose_estimator::publish_odometry(
    const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pub) {
  odomAftMapped.header.frame_id = "world";
  odomAftMapped.child_frame_id = "base";
  odomAftMapped.header.stamp = this->node->now();

  static std::shared_ptr<tf2_ros::TransformBroadcaster> br;
  br = std::make_shared<tf2_ros::TransformBroadcaster>(this->node);
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = odomAftMapped.header.stamp;
  transform.header.frame_id = "map";
  transform.child_frame_id = "base";
  transform.transform.translation.x = odomAftMapped.pose.pose.position.x;
  transform.transform.translation.y = odomAftMapped.pose.pose.position.y;
  transform.transform.translation.z = odomAftMapped.pose.pose.position.z;
  transform.transform.rotation = odomAftMapped.pose.pose.orientation;
  br->sendTransform(transform);

  static std::shared_ptr<tf2_ros::StaticTransformBroadcaster> br_static;
  br_static = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this->node);
  geometry_msgs::msg::TransformStamped static_transform;
  static_transform.header.stamp = odomAftMapped.header.stamp;
  static_transform.header.frame_id = "base";
  static_transform.child_frame_id = "base_link";
  static_transform.transform.translation.x = 0;
  static_transform.transform.translation.y = 0;
  static_transform.transform.translation.z = 0;
  static_transform.transform.rotation.x = 1;
  static_transform.transform.rotation.y = 0;
  static_transform.transform.rotation.z = 0;
  static_transform.transform.rotation.w = 0;
  br_static->sendTransform(static_transform);
  pub->publish(odomAftMapped);
}
void pose_estimator::publish_path(
    const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr &pub) {
  msg_body_pose.header.frame_id = "world";
  msg_body_pose.header.stamp = this->node->now();
  msg_body_pose.pose.position.x = odomAftMapped.pose.pose.position.x;
  msg_body_pose.pose.position.y = odomAftMapped.pose.pose.position.y;
  msg_body_pose.pose.position.z = odomAftMapped.pose.pose.position.z;
  msg_body_pose.pose.orientation = odomAftMapped.pose.pose.orientation;

  path.header.frame_id = "world";
  path.header.stamp = this->node->now();
  path.poses.clear();
  path.poses.push_back(msg_body_pose);
  pub->publish(path);
}
