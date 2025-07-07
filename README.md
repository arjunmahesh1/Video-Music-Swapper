# Video-Music-Swapper
Swapping out music within an uploaded video

Save audios in /audio, save video to swap in /video

USAGE:
  Single pair

  python swap_audio.py --video video/myclip.mp4 --audio "audio/my song.mp3" --output out.mp4


  Batch‑process every video×audio permutation in the default folders
  python swap_audio.py


  Can point to any folder/filename

  python swap_audio.py --video C:/stuff/v1.mp4 --audio C:/stuff/sound.wav



>>> out.mp4