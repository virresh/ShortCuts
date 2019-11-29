import numpy as np
import cv2
from tqdm import tqdm

"""
Memory and Time efficient video cut extractor

Tested with both short and long videos (mp4 format)
Processing time for ~3 min video = 10s, max memory usage = 0.2 Gb
Processing time for ~1.5 hr video = 2minutes, max memory usage = 0.2 Gb

(Tested on a daily use consumer laptop, hp-au0015tx)

mkv files still take longer (upto 40-45 minutes for a 100 minute video).

Video Summaries are decent on most videos.
"""


cap = cv2.VideoCapture('sample.mp4')   # This needs to be changed as per requirement

# Some useful constants
fps = 30        # the frames per second we want to keep
keep_dur = 3    # duration of clip summaries
compressed_max_dur = 60*fps    # max duration of our clip

# Some variables that will be used throughout
fnum = 0
prev = None
diff = []
frame_point = []
frame_std = []

tframe = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
if tframe < 10*60*fps:
    fskip = 1
else:
    fskip = int(max(1, tframe / compressed_max_dur))
print('Max Frames', tframe, 'Fskip', fskip)

# Extract frame numbers that have a particularly high difference between the scenes
for fitnum in tqdm(range(0, tframe, fskip)):
    fnum = fitnum
    # print('At frame', fnum)

    # Capture frame-by-frame if video is small enough
    if fskip > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
    ret, frame = cap.read()
    if not ret:
        break

    # Frame ops
    # Could convert to Grayscale as well. Though hsv gives nice intensity values
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), (32, 32))
    if prev is not None:
        s = np.std(gray)
        if s > 70:
            # print(s)
            diff.append(gray - prev)
            frame_point.append(fnum)
    prev = gray
    fnum += fskip

# Compute time-varying values
diff = np.array(diff)
mvals = np.mean(diff.reshape(diff.shape[0], -1), axis=1)
mu = np.mean(mvals)
std = np.std(mvals)

# Compute candidate frames based on mean intensity value
# I've assumed a gaussian distribution for intensity values
# and all frames lying on the extreme of bell curve are supposed to be interesting
# (within mu+1.5sigma - infinity)
candidates = []
for i, val in enumerate(mvals):
    if val > mu + 1.5 * std:
        if i + keep_dur*fps <= fnum:
            candidates.append((frame_point[i], frame_point[i]+keep_dur*fps))
        else:
            candidates.append((frame_point[i], fnum))

print('Total Candidates:', len(candidates))

del diff, mvals   # Free up unnecessary variables

# Cluster together closeby frames and compute actual playback points
playback_points = []
for i, candidate in enumerate(candidates):
    if len(playback_points) == 0:
        playback_points.append(candidate)
    else:
        last = playback_points[-1]
        if last[-1] - candidate[0] > -1 * fps:
            playback_points[-1] = (last[0], candidate[-1])
        else:
            playback_points.append(candidate)

del candidates


# Finally play the video back!
final_len = 0
for i, playback in enumerate(playback_points):
    # print('Playing Back:', i)
    cap.set(cv2.CAP_PROP_POS_FRAMES, playback[0])
    for fnum in range(*playback):
        final_len += 1
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Summary', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Print some statistics on how did the compression go
print('Compressed Video. Total Frames', fnum, 'Compressed Frames', final_len)
print('% Shortened', (100 - final_len / fnum * 100))
