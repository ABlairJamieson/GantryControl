#!/usr/bin/env python
######## Program to move x,y,z axes by desired mm. ###########
######## Just answer the questions the program asks        ###########

import gantrycontrol as gc

gantry = gc.gantrycontrol()

#Home gantry and quit
gantry.locate_home_xyz()

