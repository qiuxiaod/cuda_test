rm -r *.o *.ptx *.sass

rep_list=(1 2 4 8 16 32 64)
device_id=4
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
        CUDA_VISIBLE_DEVICES=${device_id} ./main_mode0_$rep.o
    done

    # Compile the CUDA program with the current value of rep
    echo "============================================================================="
    echo "=========== Test tcgen05.ld+ld.sync.aligned.32x32b.x${rep} =================="
    echo "============================================================================="
    nvcc -arch=sm_100a -Xptxas -O3 -DREP=$rep -DTEST_MODE=3 -o main_mode3_$rep.o main.cu
    cuobjdump --dump-ptx main_mode3_$rep.o > main_mode3_$rep.ptx
    cuobjdump --dump-sass main_mode3_$rep.o > main_mode3_$rep.sass
    for itr in {0..7}
    do
        CUDA_VISIBLE_DEVICES=${device_id} ./main_mode3_$rep.o
    done

    # Compile the CUDA program with the current value of rep
    echo "============================================================================="
    echo "============= Test tcgen05.st+ld.sync.aligned.32x32b.x${rep} ===================="
    echo "============================================================================="
    nvcc -arch=sm_100a -Xptxas -O3 -DREP=$rep -DTEST_MODE=1 -o main_mode1_$rep.o main.cu
    cuobjdump --dump-ptx main_mode1_$rep.o > main_mode1_$rep.ptx
    cuobjdump --dump-sass main_mode1_$rep.o > main_mode1_$rep.sass
    for itr in {0..7}
    do
        CUDA_VISIBLE_DEVICES=${device_id} ./main_mode1_$rep.o
    done

    # Compile the CUDA program with the current value of rep
    echo "============================================================================="
    echo "=========== Test tcgen05.st+ld+ld.sync.aligned.32x32b.x${rep} =================="
    echo "============================================================================="
    nvcc -arch=sm_100a -Xptxas -O3 -DREP=$rep -DTEST_MODE=2 -o main_mode2_$rep.o main.cu
    cuobjdump --dump-ptx main_mode2_$rep.o > main_mode2_$rep.ptx
    cuobjdump --dump-sass main_mode2_$rep.o > main_mode2_$rep.sass
    for itr in {0..7}
    do
        CUDA_VISIBLE_DEVICES=${device_id} ./main_mode2_$rep.o
    done

    # Compile the CUDA program with the current value of rep
    echo "============================================================================="
    echo "=========== Test fpu + tcgen05.ld+ld.sync.aligned.32x32b.x${rep} =================="
    echo "============================================================================="
    nvcc -arch=sm_100a -Xptxas -O3 -DREP=$rep -DTEST_MODE=4 -DTHD_NUM=256 -o main_mode4_$rep.o main.cu
    cuobjdump --dump-ptx main_mode4_$rep.o > main_mode4_$rep.ptx
    cuobjdump --dump-sass main_mode4_$rep.o > main_mode4_$rep.sass
    for itr in {0..7}
    do
        CUDA_VISIBLE_DEVICES=${device_id} ./main_mode4_$rep.o
    done

    # Compile the CUDA program with the current value of rep
    echo "============================================================================="
    echo "=========== Test AMMA + tcgen05.ld+ld.sync.aligned.32x32b.x${rep} =================="
    echo "============================================================================="
    nvcc -arch=sm_100a -Xptxas -O3 -DREP=$rep -DTEST_MODE=5 -o main_mode5_$rep.o main.cu
    cuobjdump --dump-ptx main_mode5_$rep.o > main_mode5_$rep.ptx
    cuobjdump --dump-sass main_mode5_$rep.o > main_mode5_$rep.sass
    for itr in {0..7}
    do
        CUDA_VISIBLE_DEVICES=${device_id} ./main_mode5_$rep.o
    done

    echo  "  "
    echo  "  "
done
