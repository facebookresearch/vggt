
mkdir -p /mimer/NOBACKUP/groups/3d-dl/eth3d
cd /mimer/NOBACKUP/groups/3d-dl/eth3d

wget https://www.eth3d.net/data/multi_view_training_dslr_jpg.7z
# install 7zip or p7zip on your system if not already installed
7z x multi_view_training_dslr_jpg.7z -bsp1
rm multi_view_training_dslr_jpg.7z

scenes=("courtyard" "delivery_area" "electro" "facade" "kicker" "meadow" "office" "pipes" "playground" "relief" "relief_2" "terrace" "terrains")
for scene in "${scenes[@]}"; do
    wget -c https://www.eth3d.net/data/${scene}_dslr_depth.7z
    7z x ${scene}_dslr_depth.7z -bsp1
    rm ${scene}_dslr_depth.7z
done

cd /mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt

# python training/data/preprocess/prepare_eth3d.py