for i in p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15; do
    (echo "pnet_remote('server')" | ssh -x $i matlab -nojvm -nodisplay) &
done

echo -n "Hit Return to kill remote matlab shells..."
read

for i in p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15; do
    ssh -x $i killall matlab
done
