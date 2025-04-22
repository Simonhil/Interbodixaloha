#!/bin/bash

# Check if a device path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 /dev/videoX"
    exit 1
fi

DEVICE="$1"

# Set resolution and pixel format
v4l2-ctl -d "$DEVICE" --set-fmt-video=width=1280,height=720,pixelformat=MJPG

# Set frame rate
v4l2-ctl -d "$DEVICE" --set-parm=60

#brightness
v4l2-ctl -d "$DEVICE" --set-ctrl=brightness=140
v4l2-ctl -d "$DEVICE" --set-ctrl=brightness=150



#contrasat  contrast
v4l2-ctl -d "$DEVICE" --set-ctrl=contrast=160
v4l2-ctl -d "$DEVICE" --set-ctrl=contrast=170

# Configure exposure settings
v4l2-ctl -d "$DEVICE" --set-ctrl=auto_exposure=1                 # Manual mode
v4l2-ctl -d "$DEVICE" --set-ctrl=exposure_time_absolute=208     # Adjust as needed
v4l2-ctl -d "$DEVICE" --set-ctrl=exposure_dynamic_framerate=0    # Disable dynamic frame rate

# Set gain
v4l2-ctl -d "$DEVICE" --set-ctrl=gain=5                          # Adjust as needed

# Configure white balance
v4l2-ctl -d "$DEVICE" --set-ctrl=white_balance_automatic=0
v4l2-ctl -d "$DEVICE" --set-ctrl=white_balance_temperature=4000  # Adjust as needed

# Set power line frequency (adjust based on your region)
v4l2-ctl -d "$DEVICE" --set-ctrl=power_line_frequency=1          # 1: 50 Hz, 2: 60 Hz

echo "Camera settings applied to $DEVICE"