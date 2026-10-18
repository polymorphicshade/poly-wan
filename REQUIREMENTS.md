# Overview
This repo is meant to be a custom implementation of Wan Video: https://github.com/Wan-Video/Wan2.1

Use Python, Torch, FastAPI, etc to complete the implementation.

# Requirements
1. Core Functionality
   1. generate a video with a text prompt
   2. generate a video with a reference image and an optional text prompt
   3. advanced motion control (i.e. with Wan 2.1 Control / VACE)
   4. pose-guided driving & motion-transfer with human pose preprocessors (like OpenPose or DWPose)
   5. Wan-Animate / Character Animation
      - for example, user want to animate character poses, facial expressions, and gestures accurately
   6. Video-to-Video (V2V) & Video Editing
      - for example, user wants to change a video from realistic to anime-style
   7. efficent VRAM usage to support videos of any length
      - app should have the option to use a FramePack implementation to keep the VRAM complexity constant (O(1))
      - for example, the user wants to generate extended video clips (tens of seconds to minutes) on a single consumer GPU (like an 8GB–12GB card) rather than hitting an immediate Out-Of-Memory (OOM) crash
   8. abiliy to export generated videos as MP4, WEBM, for GIF
      - use FFMPEG (have it auto-download if missing) to perform the encoding/exporting
2. Web UI
   1. a simple python web UI that allows the user to easily-use this implementation
   2. should allow the user to create/close tabs
      - each tab represents a workspace with all the controls available to the user (similar to if they were to open the web UI in multiple browser tabs)
3. Web API
   1. each core functionality should have an API endpoint
   2. the web UI should implement this API
4. Terminal CLI
   1. the app should support command-line invocation (i.e. for scripting)
   2. each core functionality should be accessible through a CLI