#!/usr/bin/env python3
import pexpect
import time
import argparse
import subprocess
import os
import sys

#get output filename for all channels
#wave0.txt -> output_filename0.txt
#wave1.txt -> output_filename1.txt
#wave2.txt -> output_filename2.txt
def get_destination(source_filename, output_filename):
    base_name1, _ = os.path.splitext(source_filename) 
    number = base_name1[-1]  
    base_name2, ext2 = os.path.splitext(output_filename)
    new_file_name = f"{base_name2}{number}{ext2}"

    return new_file_name

#main program starts here
parser = argparse.ArgumentParser(
    prog=os.path.basename(sys.argv[0]),
    description='Program to take data for t seconds using wavedump',
    epilog='@Author: Tapendra B C (Nov 2024)'
)

parser.add_argument('--duration', type=float, default=10, help='Time in seconds')
parser.add_argument('-output','-o', type=str, default='', help='Output filename')

args = parser.parse_args()
duration = args.duration
output_filename = args.output


# Start the WaveDump program
daq_process = pexpect.spawn('wavedump')
# Wait for the prompt 
daq_process.expect('s] start/stop the acquisition') 


print("Data acquisition started.")
print("Taking data for ",duration, " seconds" )
daq_process.sendline('s') #Start acquisition
daq_process.sendline('W') #Start continuous write to file

# take data for duration
time.sleep(duration)

# stop acquision
daq_process.sendline('s')
# quit program
daq_process.sendline('q')

# Wait for the program to exit
daq_process.expect(pexpect.EOF)
daq_process.close()
print("Data acquisition Ended")


#if output filename is provided, move the output file to filename
if output_filename !='':
    try:
        source_file = "wave0.txt"
        destination = get_destination(source_file, output_filename)
        subprocess.run(["mv", source_file, destination], check=True)
        print(f"{source_file} moved to {destination}")

        #source_file = "wave1.txt"
        #destination = get_destination(source_file, output_filename)
        #subprocess.run(["mv", source_file, destination], check=True)
        #print(f"{source_file} moved to {destination}")

        #source_file = "wave2.txt"
        #destination = get_destination(source_file, output_filename)
        #subprocess.run(["mv", source_file, destination], check=True)
        #print(f"{source_file} moved to {destination}")

        #source_file = "wave3.txt"
        #destination = get_destination(source_file, output_filename)
        #subprocess.run(["mv", source_file, destination], check=True)
        #print(f"{source_file} moved to {destination}")

        
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while moving file: {e}")


