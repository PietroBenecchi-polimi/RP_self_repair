simulate[<=50;1]{scs, served[0],  humanPositionX[currH-1]/100, humanPositionY[currH-1]/100, robPositionX[currR-1]/100, robPositionY[currR-1]/100, dX[currR-1]/100, dY[currR-1]/100, PATH}
Pr[<=50](<> scs)
Pr[<=42](<> scs)
Pr[<=38](<> scs)
Pr[<=35](<> scs)
Pr[<=33](<> scs)
E[<=50](max:humanFatigue[0])
E[<=50](min:batteryCharge[0])
