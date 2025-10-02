#!/usr/bin/env python3
#*****************************************************************************************#
#  This Program moves the Gantry in a portion of a sphere around the LED. 
#  For this the camera parameters should be defined in parameters_led.txt file. 
#  The led location in tank is represented by r_c    
#  and is measured in mm.                    
#
#  Coordinate system definition is defined in gantry_spherical_scan.py.
#*******************************************************************************************

import datetime
import gantrycontrol as gc
from gantry_spherical_scan import camera
from gantry_spherical_scan import get_gantry_settings
import pgcamera2 as pg
import time
import subprocess
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt


deg2rad   = np.pi/180.0
rad2deg   = 180.0/np.pi

SETUP_TIME = 0 #5*60 #Use time that is multiple of 30 s
DELAY = 0
DELAY_TIME = datetime.timedelta(seconds=DELAY)
TIMELAPSE_INTERVAL = 30

#loading parameters for the scan from parameters_sphere.txt file
class Parameters:
    '''
    Default paramters are:
    Nscan     = 200
    Rscan     = 450.0 # mm
    ledpos    = [800.0, 650.0, -600.0] #mm
    ledfacing = [0.0, -1.0, 0.0]
    phimin    = -80.0 * deg2rad
    phimax    = 80.0 * deg2rad
    thetamin  = 20.0 * deg2rad
    thetamax  = 85.0 * deg2rad
    interval  = 30 #s
    '''
    def __init__(self,filename="parameters_sphere.txt"):
        self.Nscan     = 200
        self.Rscan     = 450.0 # cm
        self.ledpos    = [800.0, 650.0, -600.0] #cm
        self.ledfacing = [0.0, -1.0, 0.0]
        self.phimin    = -80.0 * deg2rad
        self.phimax    = 80.0 * deg2rad
        self.thetamin  = 20.0 * deg2rad
        self.thetamax  = 85.0 * deg2rad
        self.interval = 30 #s
        self.load_parameters(filename)
        self.print_parameters()

    def load_parameters(self,filename):
        try:
            f=open(filename,"r")
            for line in f:
                line_s=line.split()
                first_char=line_s[0][0]
                if first_char != '#': #Adding capability to add comments. Lines starting with '#' in parameters.txt are skipped.
                    name=line_s[0]
                    if name=="ledpos":
                        val=line.split('=')[1].split('#')[0].split(',')
                        val=[float(i) for i in val]
                        self.ledpos=np.array(val)
                    if name=="ledfacing":
                        val=line.split('=')[1].split('#')[0].split(',')
                        val=[float(i) for i in val]
                        self.ledfacing=np.array(val)
                    elif name=="phimin":
                        val=line.split('=')[1].split('#')[0]
                        self.phimin=deg2rad*float(val)
                    elif name=="phimax":
                        val=line.split('=')[1].split('#')[0]
                        self.phimax=deg2rad*float(val)
                    elif name=="Nscan":
                        val=line.split('=')[1].split('#')[0]
                        self.Nscan = int(val)
                    elif name=="thetamin":
                        val=line.split('=')[1].split('#')[0]
                        self.thetamin = deg2rad*float(val)
                    elif name=="thetamax":
                        val=line.split('=')[1].split('#')[0]
                        self.thetamax = deg2rad*float(val)
                    elif name=="Rscan":
                        val=line.split('=')[1].split('#')[0]
                        self.Rscan = float(val)
                    elif name=="interval":
                        val=line.split('=')[1].split('#')[0]
                        self.interval = int(val)
                print(first_char)
            f.close()
        except FileNotFoundError:
            print("Oops! can't find "+filename)
        except ValueError:
            print("Oops!  Some parameter value isn't a number.")


    def print_parameters(self):
        print('Nscan    =', self.Nscan)
        print('Rscan    =', self.Rscan, 'mm')
        print('ledpos   =',self.ledpos, 'mm')
        print('ledfacing=',self.ledfacing)
        print('phimin   =',rad2deg*self.phimin,'deg')
        print('phimax   =',rad2deg*self.phimax,'deg')
        print('thetamin =',rad2deg*self.thetamin,'deg')
        print('thetamax =',rad2deg*self.thetamax,'deg')
        print('interval =',self.interval,'s')

def plot_scan(c1, gsets, tls):
    npgsets = np.array(gsets)
    GX, GY, GZ, Gphi, Gtheta = npgsets.T

    nptls = np.array(tls)
    # Target offset position relative to gantry
    TX, TY, TZ = nptls[:,0], nptls[:,1], nptls[:,2]
    # Target pointing direction (normalized)
    NX, NY, NZ = nptls[:,3], nptls[:,4], nptls[:,5]

    steps = np.arange(len(GX))

    # --- Gantry position plot ---
    fig = plt.figure(figsize=(5,5), dpi=120)
    ax = fig.add_subplot(projection='3d')
    ax.plot(GX, GY, GZ, color='lightgray', alpha=0.5)  # path in light gray
    sc = ax.scatter(GX, GY, GZ, c=steps, cmap='viridis', marker='o', s=20)
    ax.scatter(c1.rc[0], c1.rc[1], c1.rc[2], marker='*', color='red', s=100)
    ax.set_xlabel('xg (mm)')
    ax.set_ylabel('yg (mm)')
    ax.set_zlabel('zg (mm)')
    ax.set_title('Gantry position')
    fig.colorbar(sc, ax=ax, label='Step number')
    plt.savefig('gantrypos_led.png')

    # --- Target position and pointing plot ---
    fig = plt.figure(figsize=(5,5), dpi=120)
    ax = fig.add_subplot(projection='3d')
    # Arrow length can be reduced for clarity if many points
    arrow_length = 30.0
    arrow_width = 2.0

    for i in range(len(GX)):
        ax.quiver(GX[i]+TX[i], GY[i]+TY[i], GZ[i]+TZ[i],
                  NX[i], NY[i], NZ[i],
                  length=arrow_length, normalize=True,
                  color=cmap(steps[i]/len(GX)), linewidth=arrow_width)

    sc = ax.scatter(GX+TX, GY+TY, GZ+TZ, c=steps, cmap=cmap, marker='o', s=20)
    ax.scatter(c1.rc[0], c1.rc[1], c1.rc[2], marker='*', color='red', s=100)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.set_title('Target position and pointing')
    fig.colorbar(sc, ax=ax, label='Step number')
    plt.tight_layout()
    plt.savefig('gantrypospnt_led.png')

    print('min Gx=', np.min(GX), ' max Gx=', np.max(GX))
    print('min Gy=', np.min(GY), ' max Gy=', np.max(GY))
    print('min Gz=', np.min(GZ), ' max Gz=', np.max(GZ))


# =====================================================================================
def main():
    parser = argparse.ArgumentParser( description='scan_led_spherical options' )
    parser.add_argument('-p','--param_file',default='parameters_led.txt', help='Parameter file')
    parser.add_argument('--dryrun',help='Do dry run, printing scan locations only',action='store_true')
    parser.add_argument('--no-dryrun',dest='dryrun',action='store_false')
    parser.set_defaults(dryrun=True)

    
    args = parser.parse_args()
    print(args)
    param  = Parameters( args.param_file )
    cam    = camera( param.ledpos, param.ledfacing ) # bad name (sorry)
    scanpts = cam.get_scanpoints( param.Nscan, param.Rscan, param.phimin, param.phimax, param.thetamin, param.thetamax  )
    sensor_offset = np.array([50.0, 50.0, 0.0])          # along gantry x-axis and y-axis
    world_to_gantry_matrix = np.eye(3)                  # identity if axes align

    gsets, tls = get_gantry_settings( cam, scanpts, sensor_offset, world_to_gantry_matrix )

    print('Dryrun=',args.dryrun)
    if args.dryrun == True:
        print('Dry Run')
        print('gsets=')
        print( gsets )
        print('tls=')
        print(tls)
        plot_scan( cam, gsets, tls )
        return 0

    start_time = time.time()
    time.sleep(SETUP_TIME)

    gantry = gc.gantrycontrol()
    gantry.locate_home_xyz();
    #_ = input("Press enter to move")
    done_setup_time = time.time()
    print('time to setup was',done_setup_time-start_time,'s')
    for n,gset in enumerate(gsets):
        print('move',n,'to',gset)
        cur_move_start = time.time()
        curx, cury, curz, curphi, curtheta = gset
        curphi*=rad2deg
        curtheta*=rad2deg
        
        gantry.move( "DM", "DM", curz )
        gantry.move( curx, cury,"DM",curphi,curtheta,1000,1000,1000,200,200)
        output_filename = ""+str(curphi)+"_"+str(curtheta)+".txt"
        cur_move_done = time.time()
        time.sleep(2) # sleep 2 second to settle any vibration
        print('move',n,'done after',cur_move_done-cur_move_start,'s')
        subprocess.run(["wave_recorder.py","--duration", str(param.interval), "-o", output_filename])
        cur_collect_done = time.time()
        print('done collecting, point',n,'took',cur_collect_done-cur_move_start,'s')
        
    print('Done')

if __name__ == "__main__":
    sys.exit(main())
    
