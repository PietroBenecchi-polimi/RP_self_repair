simulate[<=150;1]{scs, served[0],  humanPositionX[currH-1]/100, humanPositionY[currH-1]/100, robPositionX[currR-1]/100, robPositionY[currR-1]/100, dX[currR-1]/100, dY[currR-1]/100, PATH}
Pr[<=150](<> scs)
E[<=150](max:humanFatigue[0])
E[<=150](min:batteryCharge[0])
