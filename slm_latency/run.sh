rm -r *.o *.ptx *.sass
nvcc -arch=sm_100a -Xptxas -O3 -o slm_test.o slm_test.cu
cuobjdump --dump-ptx slm_test.o > slm_test.ptx
cuobjdump --dump-sass slm_test.o > slm_test.sass

for device in {0..7}
do
    echo "Running slm_test.o on CUDA device $device"
    CUDA_VISIBLE_DEVICES=$device ./slm_test.o
done