mkdir vkitti
cd vkitti

wget https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar
tar -xvf vkitti_2.0.3_rgb.tar

wget https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_depth.tar
tar -xvf vkitti_2.0.3_depth.tar

wget https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_textgt.tar.gz
tar -xvf vkitti_2.0.3_textgt.tar.gz


cd ..

