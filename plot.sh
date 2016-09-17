gnuplot --persist << FIN
set key left top
plot "plot.data" using 1:2, "plot.data" using 1:3
exit
FIN
