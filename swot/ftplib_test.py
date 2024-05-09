import requests
import os
import ftplib
import getpass


def ftp_data_access(ftp_path, filename):
    # Set up FTP server details
    ftpAVISO = 'ftp-access.aviso.altimetry.fr'

    try:
        # Prompt for username and password
        username = input("Enter username: ")
        password = getpass.getpass(prompt=f"Enter password for {username}: ")

        # Logging into FTP server using provided credentials
        with ftplib.FTP(ftpAVISO) as ftp:
            ftp.login(username, password)
            ftp.cwd(ftp_path)
            print(f"Connection Established {ftp.getwelcome()}")

            # Check if the file exists in the directory
            if filename in ftp.nlst():
                local_filepath = input("Enter the local directory to save the file: ")
                download_file_from_ftp(ftp, filename, local_filepath)
            else:
                print(f"File {filename} does not exist in the directory.")
    except ftplib.error_perm as e:
        print(f"FTP error: {e}")
    except Exception as e:
        print(f"Error: {e}")

def download_file_from_ftp(ftp, filename, target_directory):
    try:
        local_filepath = os.path.join(target_directory, filename)
        with open(local_filepath, 'wb') as file:
            ftp.retrbinary('RETR %s' % filename, file.write)
            print(f"Downloaded {filename} to {local_filepath}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

# Define directories
ftp_path = '/swot_products/l3_karin_nadir/l3_lr_ssh/v1_0/Expert/cycle_009/'
filename = 'SWOT_L3_LR_SSH_Expert_009_583_20240124T223638_20240124T232805_v1.0.nc'

# FTP download
ftp_data_access(ftp_path, filename)
