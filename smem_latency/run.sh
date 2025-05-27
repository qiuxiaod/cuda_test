rm -r *.o *.ptx *.sass
nvcc -arch=sm_100a -Xptxas -O3 -o main.o main.cu
cuobjdump --dump-ptx main.o > main.ptx
cuobjdump --dump-sass main.o > main.sass

for device in {0..7}
do
    echo "Running main.o on CUDA device $device"
    CUDA_VISIBLE_DEVICES=$device ./main.o
done