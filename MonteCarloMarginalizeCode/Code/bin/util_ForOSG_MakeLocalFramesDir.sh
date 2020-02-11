#! /bin/bash
#
#  util_ForOSG_MakeLocalFramesDir.sh local.cache
#   



mkdir frames_dir
cat $1 | awk '{print $NF}' > my_temp_files
switcheroo file://localhost ' ' my_temp_files
for name in `cat my_temp_files`
do
  cp ${name} frames_dir
done
