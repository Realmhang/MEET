# idx=(1 2 3 4)
# for i in "${idx[@]}"; do
#     echo "Processing dataset CPSea2.0_AFDB_$i"
#     python -m scripts.hier_index.pre_shuffle --mmap_dir /mnt/data/share/eptv2/data/CPSea2_all/CPSea2.0_AFDB_$i/processed/ 
# done
python -m scripts.hier_index.pre_shuffle --mmap_dirs /mnt/data/share/eptv2/data/CPSea2_all/CPSea2.0_AFDB_1/processed/ /mnt/data/share/eptv2/data/CPSea2_all/CPSea2.0_AFDB_2/processed/ /mnt/data/share/eptv2/data/CPSea2_all/CPSea2.0_AFDB_3/processed/ /mnt/data/share/eptv2/data/CPSea2_all/CPSea2.0_AFDB_4/processed/ --output_dir /mnt/data/share/eptv2/data/CPSea2_all/