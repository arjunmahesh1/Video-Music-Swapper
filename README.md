_Motivation_: Was watching an ad, and I realized I didn't skip it and watched the whole ad because I liked the song. Thought to myself, in this day of hyper-tuned personalization, why isn't there a better and more certain way of ad engagement?

# Video-Music-Swapper
Swap any video’s soundtrack with a new audio track in one command or via a one‑page Streamlit demo.

Future:

- Spotify API integration
- Audio optimization




# RUN:
Either clone repo - or download app.py + swap_audio and install ffmpeg/streamlit

## GUI APP (easiest):
> pip install streamlit

> streamlit run app.py



## CLI USAGE:

Save audios in /audio, save video to swap in /video

> pip install ffmpeg

  _Single pair_:

  > python swap_audio.py --video video/myclip.mp4 --audio "audio/my song.mp3" --output out.mp4



  _Batch‑process every video×audio permutation in the default folders_:

  > python swap_audio.py


  Can point to any folder/filename

  > python swap_audio.py --video C:/stuff/v1.mp4 --audio C:/stuff/sound.wav

-> out.mp4





#### Advanced

The CLI has --videos-dir and --audio-dir flags to point batch mode anywhere.

To fade the new track out, edit swap_audio.py and add an audio‑filter: -af "afade=t=out:st=<sec>:d=2".

Re‑encode video for wider compatibility by swapping -c:v copy for e.g. -c:v libx264 -crf 20.