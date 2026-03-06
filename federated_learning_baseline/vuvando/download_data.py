import wfdb
import os

print("Starting MIT-BIH Download...")
print("This may take a few minutes depending on your internet.")

# Download to a folder named 'mitdb' in the current directory
wfdb.dl_database('mitdb', 'mitdb')

print("Download Complete! You can now run the simulation.")