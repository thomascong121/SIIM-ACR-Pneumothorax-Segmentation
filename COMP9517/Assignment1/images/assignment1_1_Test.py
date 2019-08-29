import sys
from assignment1 import *

#uncomment the one which you want that sepcific filtering method be applied
draw_plots(sys.argv[1],150,255,median_blur=5)
# draw_plots(sys.argv[1],150,255,GaussianBlur=[7,3,3])