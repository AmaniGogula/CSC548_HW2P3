# heatmap for lake.cu



set terminal png



set xrange[0:1]

set yrange[0:1]



set output 'lake_i.png'

plot 'lake_i.dat' using 1:2:3 with image





set output 'lake_f0.png'

plot 'lake_f_0.dat' using 1:2:3 with image



set output 'lake_f1.png'

plot 'lake_f_1.dat' using 1:2:3 with image



set output 'lake_f2.png'

plot 'lake_f_2.dat' using 1:2:3 with image



set output 'lake_f3.png'

plot 'lake_f_3.dat' using 1:2:3 with image




