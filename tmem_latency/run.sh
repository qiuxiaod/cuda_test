rm -r *.o *.ptx *.sass

rep_list=(1 2 4 8 16 32 64 128)

# Loop over each value of rep
for rep in "${rep_list[@]}"
do
    # Compile the CUDA program with the current value of rep
    echo "============================================================================="
    echo "============== Test tcgen05.ld.sync.aligned.32x32b.x${rep} ====================="
    echo "============================================================================="
    nvcc -arch=sm_100a -Xptxas -O3 -DREP=$rep -o main_$rep.o main.cu
    cuobjdump --dump-ptx main_$rep.o > main_$rep.ptx
    cuobjdump --dump-sass main_$rep.o > main_$rep.sass
    for itr in {0..7}
    do
        # echo "Running main_$rep.o on CUDA Itr$itr"
        CUDA_VISIBLE_DEVICES=4 ./main_$rep.o
    done
done


