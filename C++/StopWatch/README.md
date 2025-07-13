README for Console Stopwatch
Overview
This is a simple console-based stopwatch application written in C++. It allows users to start, pause, reset, and exit a stopwatch with elapsed time displayed in hours:minutes:seconds format along with total minutes.

Features
Real-time elapsed time display (HH:MM:SS)

Shows total minutes with decimal precision

Supports multiple commands:

Start (s)

Pause (p)

Reset (t)

Exit (x)

Help (help)

Customizable stopwatch name

Thread-safe implementation using mutex

Atomic operations for thread synchronization

Requirements
C++20 compatible compiler (uses std::format)

Standard C++ libraries

Windows (uses _CRT_SECURE_NO_WARNINGS)

Usage
Compile the program using a C++20 compatible compiler

Run the executable

Enter a name for your stopwatch when prompted

Use the following commands:

s - Start the stopwatch

p - Pause the stopwatch

t - Reset the stopwatch

x - Exit the program

help - Show available commands

Implementation Details
Uses std::chrono for high-precision timing

Separate thread for display updates

Mutex-protected time calculations

Atomic flag for running state

Formatted console output with leading zeros

Notes
The display updates every 500 milliseconds

Time is displayed in 24-hour format with +3 hour offset (adjust as needed)

Pausing preserves the elapsed time

Resetting clears all elapsed time and restarts the stopwatch

Example
text
Console Stopwatch

Enter Stopwatch name:
My Timer

Stopwatch Commands:
(s)tart - Begin timing
(p)ause - Pause the stop_watch
rese(t) - Reset to zero
e(x)it  - Quit the program

> s
My Timer Started at Time: 14:30:15
          Elapsed: 00:00:00 | Total Minutes: 0.00 mins