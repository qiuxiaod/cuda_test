rm -r *.o *.ptx *.sass

rep_list=(1 2 4 8 16 32 64)

# Loop over each value of rep
for rep in "${rep_list[@]}"
do
    # Compile the CUDA program with the current value of rep
    echo "============================================================================="
    echo "============== Test tcgen05.ld.sync.aligned.32x32b.x${rep} ====================="
    echo "============================================================================="
    nvcc -arch=sm_100a -Xptxas -O3 -DREP=$rep -DTEST_MODE=0 -o main_mode0_$rep.o main.cu
    cuobjdump --dump-ptx main_mode0_$rep.o > main_mode0_$rep.ptx
    cuobjdump --dump-sass main_mode0_$rep.o > main_mode0_$rep.sass
    for itr in {0..7}
    do
        CUDA_VISIBLE_DEVICES=3 ./main_mode0_$rep.o
    done

    # # Compile the CUDA program with the current value of rep
    # echo "============================================================================="
    # echo "============= Test tcgen05.st+ld.sync.aligned.32x32b.x${rep} ===================="
    # echo "============================================================================="
    # nvcc -arch=sm_100a -Xptxas -O3 -DREP=$rep -DTEST_MODE=1 -o main_mode1_$rep.o main.cu
    # cuobjdump --dump-ptx main_mode1_$rep.o > main_mode1_$rep.ptx
    # cuobjdump --dump-sass main_mode1_$rep.o > main_mode1_$rep.sass
    # for itr in {0..7}
    # do
    #     CUDA_VISIBLE_DEVICES=3 ./main_mode1_$rep.o
    # done

    # Compile the CUDA program with the current value of rep
    echo "============================================================================="
    echo "=========== Test tcgen05.ld+st.sync.aligned.32x32b.x${rep} =================="
    echo "============================================================================="
    nvcc -arch=sm_100a -Xptxas -O3 -DREP=$rep -DTEST_MODE=2 -o main_mode2_$rep.o main.cu
    cuobjdump --dump-ptx main_mode2_$rep.o > main_mode2_$rep.ptx
    cuobjdump --dump-sass main_mode2_$rep.o > main_mode2_$rep.sass
    for itr in {0..7}
    do
        CUDA_VISIBLE_DEVICES=3 ./main_mode2_$rep.o
    done
    echo  "  "
    echo  "  "
done
