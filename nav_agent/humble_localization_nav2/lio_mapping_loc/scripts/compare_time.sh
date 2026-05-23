# python3 compare_run_time.py /home/users/tingyang.xiao/VLN/Log/reloc_time_pcl_icp.txt \
#  /home/users/tingyang.xiao/VLN/Log/reloc_time_pcl_ndt.txt \
#  /home/users/tingyang.xiao/VLN/Log/reloc_time_gpu_ndt.txt \
#   -o /home/users/tingyang.xiao/VLN/Log/compare_time --no-show
python3 compare_run_time.py \
 /Log/result/reloc_time_pcl_ndt_0.05.txt \
 /Log/result/reloc_time_pcl_icp_0.05.txt \
 /Log/result/reloc_time_gpu_ndt_0.05.txt \
  -o /home/users/tingyang.xiao/VLN/Log/compare_time_dense0 --no-show