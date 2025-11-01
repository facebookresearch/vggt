# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

mkdir vkitti
cd vkitti

wget https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar
tar -xvf vkitti_2.0.3_rgb.tar

wget https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_depth.tar
tar -xvf vkitti_2.0.3_depth.tar

wget https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_textgt.tar.gz
tar -xvf vkitti_2.0.3_textgt.tar.gz


cd ..

