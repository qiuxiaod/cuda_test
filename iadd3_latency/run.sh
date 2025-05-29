rm -r *.o *.ptx *.sass

rep_list=(9  11  13 15)

# Loop over each value of rep
for rep in "${rep_list[@]}"
do
    # Compile the CUDA program with the current value of rep
    iadd_num=$((rep / 2))
    echo "============================================================================="
    echo "========================== Test IADD3 b2b x${iadd_num} ==============================="
    echo "============================================================================="
    nvcc -arch=sm_100a -Xptxas -O3 -DREP=$rep -o main_$iadd_num.o main.cu
    cuobjdump --dump-ptx main_$iadd_num.o > main_$iadd_num.ptx
    cuobjdump --dump-sass main_$iadd_num.o > main_$iadd_num.sass
    for itr in {0..7}
    do
        CUDA_VISIBLE_DEVICES=7 ./main_$iadd_num.o
    done

done
